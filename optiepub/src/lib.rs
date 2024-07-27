use argh::FromArgs;
use eyre::Result;
use image::imageops::FilterType::Lanczos3;
use image::{DynamicImage, GenericImageView, ImageReader, Rgba};
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Cursor, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::LazyLock;
use zip::write::SimpleFileOptions;
use zip::{ZipArchive, ZipWriter};

static FIGURE_REF_REGEX: LazyLock<Regex> =
	LazyLock::new(|| Regex::new(r#"(?i)<svg[^>]*?><image\s+?xlink:href="([^"<>]+?)".+?</svg>"#).unwrap());

static FILE_REFS_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"(?i)src\s*=\s*"([^"<>]+?)""#).unwrap());

#[derive(FromArgs)]
/// Optimize EPUB files by compressing images and removing unnecessary content.
#[derive(Default)]
pub struct Cli {
	/// input EPUB file path
	#[argh(positional)]
	input: PathBuf,

	/// output EPUB file path
	#[argh(option, short = 'o')]
	output: Option<PathBuf>,

	/// JPEG quality (1-100)
	#[argh(option, short = 'q', default = "75")]
	jpeg_quality: u8,

	/// maximum image dimension
	#[argh(option, short = 'd', default = "1440")]
	max_dimension: u32,

	/// hash distance for considering images similar
	#[argh(option, short = 'h', default = "6")]
	hash_distance: u32,

	/// log more info when processing epubs
	#[argh(switch, short = 'v')]
	verbose: bool,

	/// show a progress bar
	#[argh(switch, short = 'p')]
	no_progress: bool,
}

#[derive(Default)]
struct Statistics {
	total_images: usize,
	optimized_images: usize,
	removed_unused: usize,
	removed_duplicate: usize,
	original_size: u64,
	optimized_size: u64,
}

impl Statistics {
	fn percentage_saved(&self) -> f32 {
		if self.original_size == 0 {
			0.0
		} else {
			(1.0 - (self.optimized_size as f32 / self.original_size as f32)) * 100.0
		}
	}
}

struct ImmutableState<'a> {
	cli: &'a Cli,
	image_paths: HashMap<String, String>,
	progress_bar: Option<ProgressBar>,
}

struct MutableState<ZW: io::Read + io::Seek + io::Write = File> {
	zip: ZipArchive<ZW>,
	outzip: ZipWriter<ZW>,
	image_hashes: HashMap<String, (ImageHash, ImageHash)>,
	optimized_images: HashMap<String, String>,
	stats: Statistics,
}

fn comp_jpeg(image: DynamicImage, quality: f32) -> Result<Vec<u8>> {
	let mut comp = mozjpeg::Compress::new(mozjpeg::ColorSpace::JCS_RGB);
	comp.set_quality(quality);
	comp.set_size(image.width() as _, image.height() as _);
	let mut comp = comp.start_compress(Vec::new())?;
	comp.write_scanlines(&image.into_rgb8())?;
	Ok(comp.finish()?)
}

fn needs_alpha(image: &DynamicImage) -> bool {
	if !image.color().has_alpha() {
		return false;
	}
	if let Some(argb8) = image.as_rgba8() {
		let count = argb8.pixels().filter(|&pixel| pixel.0[3] < 200).count();
		return count > argb8.pixels().len() / 20;
	}
	false
}

use image_hasher::{HasherConfig, ImageHash};
use imageproc::edges::canny;

fn crop_transparent_and_black(img: DynamicImage) -> DynamicImage {
	let (width, height) = img.dimensions();
	let gray_img = img.to_luma8();
	let edges = canny(&gray_img, 50.0, 100.0);

	let is_content = |pixel: Rgba<u8>| -> bool { pixel.0[3] > 10 || pixel.0[0] > 10 || pixel.0[1] > 10 || pixel.0[2] > 10 };

	let content_density = |x: u32, y: u32, w: u32, h: u32| -> f32 {
		let mut content_count = 0;
		let mut edge_count = 0;
		let mut total = 0;

		for dy in 0..h {
			for dx in 0..w {
				if x + dx < width && y + dy < height {
					total += 1;
					if is_content(img.get_pixel(x + dx, y + dy)) {
						content_count += 1;
					}
					if edges.get_pixel(x + dx, y + dy).0[0] > 0 {
						edge_count += 1;
					}
				}
			}
		}

		if total == 0 {
			return 0.0;
		}

		(content_count as f32 / total as f32) * 0.7 + (edge_count as f32 / total as f32) * 0.3
	};

	let mut left = width;
	let mut right = 0;
	let mut top = height;
	let mut bottom = 0;

	let window_size = 20;
	let step_size = 10;

	for y in (0..height).step_by(step_size as usize) {
		for x in (0..width).step_by(step_size as usize) {
			if content_density(x, y, window_size, window_size) > 0.05 {
				left = left.min(x);
				right = right.max(x + window_size - 1);
				top = top.min(y);
				bottom = bottom.max(y + window_size - 1);
			}
		}
	}

	macro_rules! density_check {
		($i:expr, $is_horizontal:expr) => {{
			if if $is_horizontal {
				content_density($i, 0, 1, height)
			} else {
				content_density(0, $i, width, 1)
			} > 0.01
			{
				return $i;
			}
		}};
	}

	let refine_edge = {
		#[inline(always)]
		|start: u32, end: u32, max: u32, is_horizontal: bool, is_forward: bool| {
			if is_forward {
				for i in start..=end {
					density_check!(i, is_horizontal);
				}
				0
			} else {
				for i in (start..=end).rev() {
					density_check!(i, is_horizontal);
				}
				max
			}
		}
	};

	let (left, right, top, bottom) = (
		refine_edge(left, right, width, true, true),
		refine_edge(left, right, width, true, false),
		refine_edge(top, bottom, height, false, true),
		refine_edge(top, bottom, height, false, false),
	);

	if left < right && top < bottom && right < width && bottom < height {
		img.crop_imm(left, top, right - left + 1, bottom - top + 1)
	} else {
		img
	}
}

fn optimize_image(cli: &Cli, name: &str, image: &[u8], max_dim: u32, jpeg_quality: f32) -> Result<(&'static str, Vec<u8>, ImageHash)> {
	if cli.verbose {
		eprintln!("Optimizing image: {}", name);
	}

	let mut img_rs = ImageReader::new(Cursor::new(image)).with_guessed_format()?.decode()?;

	if cli.verbose {
		eprintln!("Original dimensions: {}x{}", img_rs.width(), img_rs.height());
	}

	img_rs = crop_transparent_and_black(img_rs);

	if cli.verbose {
		eprintln!("Dimensions after cropping: {}x{}", img_rs.width(), img_rs.height());
	}

	if img_rs.width() > max_dim || img_rs.height() > max_dim {
		img_rs = img_rs.resize(max_dim, max_dim, Lanczos3);
		if cli.verbose {
			eprintln!("Resized to: {}x{}", img_rs.width(), img_rs.height());
		}
	}

	if !needs_alpha(&img_rs) {
		let img_rs = DynamicImage::from(img_rs.into_rgb8());
		let hash = calculate_image_hash_from_loaded_image(&img_rs);

		if cli.verbose {
			eprintln!("Optimizing as JPEG");
		}

		return Ok(("jpg", comp_jpeg(img_rs, jpeg_quality)?, hash));
	}

	let hash = calculate_image_hash_from_loaded_image(&img_rs);
	let mut bytes: Vec<u8> = Vec::new();
	img_rs.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)?;

	if cli.verbose {
		eprintln!("Optimizing as PNG");
	}

	Ok((
		"png",
		oxipng::optimize_from_memory(
			&bytes,
			&oxipng::Options {
				fix_errors: true,
				..Default::default()
			},
		)?,
		hash,
	))
}

fn calculate_image_hash(image_data: &[u8]) -> Result<ImageHash> {
	Ok(calculate_image_hash_from_loaded_image(&image::load_from_memory(image_data)?))
}

fn calculate_image_hash_from_loaded_image(image_data: &DynamicImage) -> ImageHash {
	let hasher = HasherConfig::new().hash_size(8, 8).preproc_dct().to_hasher();
	hasher.hash_image(image_data)
}

fn add_mimetype_file(outzip: &mut ZipWriter<File>) -> Result<()> {
	let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
	outzip.start_file("mimetype", options)?;
	outzip.write_all(b"application/epub+zip")?;
	Ok(())
}

fn collect_image_paths(zip: &mut ZipArchive<impl io::Read + io::Seek>) -> Result<HashMap<String, String>> {
	let mut image_paths = HashMap::new();
	for i in 0..zip.len() {
		let file = zip.by_index(i)?;
		if file.is_file() {
			if let Some(ext) = get_file_extension(file.name()) {
				if ["png", "webp", "jpeg", "jpg"].contains(&ext.to_ascii_lowercase().as_str()) {
					let lowercase_name = file.name().to_ascii_lowercase();
					image_paths.insert(lowercase_name, file.name().to_string());
				}
			}
		}
	}
	Ok(image_paths)
}

fn get_file_extension(name: &str) -> Option<String> {
	Path::new(name)
		.extension()
		.and_then(|ext| ext.to_str())
		.map(|ext| ext.to_ascii_lowercase())
}

pub fn optimize(cli: &Cli) -> Result<()> {
	let mut zip = ZipArchive::new(File::open(&cli.input)?)?;

	let output_path = cli.output.as_ref().cloned().unwrap_or_else(|| {
		cli.input
			.with_file_name(format!("{}_optimized.epub", cli.input.file_stem().unwrap().to_string_lossy()))
	});

	let outzip = ZipWriter::new(File::create(&output_path)?);

	let image_paths = collect_image_paths(&mut zip)?;

	let progress_bar = if cli.no_progress {
		None
	} else {
		Some(ProgressBar::new(zip.len() as u64))
	};

	if let Some(p) = &progress_bar {
		p.set_style(
			ProgressStyle::default_bar()
				.template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
				.unwrap()
				.progress_chars("##-"),
		);
	}

	let immutable_state = ImmutableState {
		cli,
		image_paths,
		progress_bar,
	};

	let mut mutable_state = MutableState {
		zip,
		outzip,
		image_hashes: HashMap::new(),
		optimized_images: HashMap::new(),
		stats: Statistics::default(),
	};

	mutable_state.stats.original_size = std::fs::metadata(&cli.input)?.len();
	mutable_state.stats.total_images = immutable_state.image_paths.len();

	add_mimetype_file(&mut mutable_state.outzip)?;

	for i in 0..mutable_state.zip.len() {
		if let Some(p) = &immutable_state.progress_bar {
			p.set_position(i as u64);
		}

		let file = mutable_state.zip.by_index(i)?;
		let name = file.name().to_owned();

		if let Some(p) = &immutable_state.progress_bar {
			p.set_message(format!("Processing {}", name));
		}

		if name == "mimetype" {
			continue;
		}

		let extension = get_file_extension(&name);

		match extension.as_deref() {
			Some("png" | "webp" | "jpeg" | "jpg") => {}
			Some("xhtml" | "html") => {
				let mut html_content = String::new();
				{
					let mut file = file;
					file.read_to_string(&mut html_content)?;
				}
				process_html_file(&immutable_state, &mut mutable_state, html_content, &name)?;
			}
			_ => {
				let mut file = file;
				copy_file_to_output(&mut file, &name, &mut mutable_state.outzip)?;
			}
		}
	}

	mutable_state.outzip.finish()?;
	mutable_state.stats.optimized_size = std::fs::metadata(&output_path)?.len();

	if let Some(p) = &immutable_state.progress_bar {
		p.finish_with_message(format!(
			"Optimization complete, saved to {}.\n\
					 Total images: {}\n\
					 Optimized images: {}\n\
					 Removed unused: {}\n\
					 Removed duplicate: {}\n\
					 Original size: {:.2} MiB\n\
					 Optimized size: {:.2} MiB\n\
					 Percentage saved: {:.2}%",
			output_path.display(),
			mutable_state.stats.total_images,
			mutable_state.stats.optimized_images,
			mutable_state.stats.removed_unused,
			mutable_state.stats.removed_duplicate,
			mutable_state.stats.original_size as f32 / (1024.0 * 1024.0),
			mutable_state.stats.optimized_size as f32 / (1024.0 * 1024.0),
			mutable_state.stats.percentage_saved(),
		));
	}

	Ok(())
}

fn process_html_file(immutable_state: &ImmutableState, mutable_state: &mut MutableState, html_content: String, name: &str) -> Result<()> {
	let html_folder = PathBuf::from(name).parent().unwrap_or(Path::new("")).to_path_buf();

	let content = replace_image_references(&FILE_REFS_REGEX, immutable_state, mutable_state, &html_content, &html_folder);
	let content = replace_image_references(&FIGURE_REF_REGEX, immutable_state, mutable_state, &content, &html_folder);

	mutable_state.outzip.start_file(
		name,
		SimpleFileOptions::default().compression_method(zip::CompressionMethod::DEFLATE),
	)?;
	mutable_state.outzip.write_all(content.as_bytes())?;

	Ok(())
}

fn replace_image_references(
	regex: &Regex,
	immutable_state: &ImmutableState,
	mutable_state: &mut MutableState<impl io::Read + io::Seek + io::Write>,
	content: &str,
	html_folder: &Path,
) -> String {
	regex
		.replace_all(content, |cap: &regex::Captures| {
			if let Some(img) = cap.get(1) {
				let resolved_path = join_path(html_folder, img.as_str());
				if let Some(img_path) = immutable_state
					.image_paths
					.get(&resolved_path.to_string_lossy().to_ascii_lowercase())
				{
					let new_path = get_or_create_optimized_image(immutable_state, mutable_state, img_path);
					let relativized = make_rel_path(html_folder, Path::new(&new_path));
					return if cap[0].to_ascii_lowercase().starts_with("<svg") {
						format!(r#"<img src="{}"/>"#, relativized.display())
					} else {
						format!(r#"src="{}""#, relativized.display())
					};
				} else {
					mutable_state.stats.removed_unused += 1;
				}
			}

			if immutable_state.cli.verbose {
				eprintln!("Skipping unknown image {} referenced in {}", &cap[0], html_folder.display());
			}
			cap[0].to_string()
		})
		.into_owned()
}

fn get_or_create_optimized_image(
	immutable_state: &ImmutableState,
	mutable_state: &mut MutableState<impl io::Read + io::Seek + io::Write>,
	img_path: &str,
) -> String {
	if let Some(optimized_path) = mutable_state.optimized_images.get(img_path) {
		return optimized_path.clone();
	}

	let mut buf = vec![];
	mutable_state.zip.by_name(img_path).unwrap().read_to_end(&mut buf).unwrap();

	let original_hash = calculate_image_hash(&buf).unwrap();

	let similar_image = mutable_state.image_hashes.iter().find(|(_, (hash1, hash2))| {
		hash1.dist(&original_hash) <= immutable_state.cli.hash_distance || hash2.dist(&original_hash) <= immutable_state.cli.hash_distance
	});

	let new_path = if let Some((existing_path, _)) = similar_image {
		mutable_state.stats.removed_duplicate += 1;
		existing_path.clone()
	} else {
		let (new_ext, res, optimized_hash) = optimize_image(
			immutable_state.cli,
			img_path,
			&buf,
			immutable_state.cli.max_dimension,
			immutable_state.cli.jpeg_quality as f32,
		)
		.unwrap();

		let similar_image = mutable_state.image_hashes.iter().find(|(_, (hash1, hash2))| {
			hash1.dist(&optimized_hash) <= immutable_state.cli.hash_distance || hash2.dist(&optimized_hash) <= immutable_state.cli.hash_distance
		});

		if let Some((existing_path, _)) = similar_image {
			mutable_state.stats.removed_duplicate += 1;
			existing_path.clone()
		} else {
			let new_path = swap_ext(img_path, new_ext);
			if immutable_state.cli.verbose {
				eprintln!("Saving optimized image to {new_path}");
			}
			mutable_state
				.outzip
				.start_file(
					new_path.to_owned(),
					SimpleFileOptions::default().compression_method(zip::CompressionMethod::DEFLATE),
				)
				.unwrap();
			mutable_state.outzip.write_all(&res).unwrap();

			mutable_state.image_hashes.insert(new_path.clone(), (original_hash, optimized_hash));
			mutable_state.stats.optimized_images += 1;
			new_path
		}
	};

	mutable_state.optimized_images.insert(img_path.to_owned(), new_path.clone());
	new_path
}

fn copy_file_to_output(file: &mut zip::read::ZipFile, name: &str, outzip: &mut ZipWriter<File>) -> Result<()> {
	outzip.start_file(
		name,
		SimpleFileOptions::default().compression_method(zip::CompressionMethod::DEFLATE),
	)?;
	std::io::copy(file, outzip)?;
	Ok(())
}

fn swap_ext(name: &str, new_ext: &str) -> String {
	let name_non_ext = &name[0..name.rfind('.').unwrap_or(name.len())];
	format!("{name_non_ext}.{new_ext}")
}

fn make_rel_path(current_folder: &Path, target: &Path) -> PathBuf {
	let current_parts: Vec<_> = current_folder.components().collect();
	let target_parts: Vec<_> = target.components().collect();

	let common_prefix = current_parts.iter().zip(&target_parts).take_while(|&(a, b)| a == b).count();

	let up_levels = current_parts.len() - common_prefix;
	let down_path = &target_parts[common_prefix..];

	std::iter::repeat(std::path::Component::ParentDir)
		.take(up_levels)
		.chain(down_path.iter().cloned())
		.collect()
}

fn join_path(path: &Path, img: &str) -> PathBuf {
	let rel_path = path.join(img);
	let mut components = Vec::new();
	for ele in rel_path.components() {
		match ele {
			Component::ParentDir => {
				components.pop();
			}
			_ => components.push(ele),
		}
	}
	components.iter().collect()
}

#[cfg(test)]
mod tests {
	use super::*;
	use imageproc::drawing::Canvas;
	use io::SeekFrom;
	use std::io::{Cursor, Seek};
	use tempfile::{tempdir, NamedTempFile};

	#[test]
	fn test_optimize_integration() -> Result<()> {
		// Create a mock EPUB file in memory
		let mut epub_data = Cursor::new(Vec::new());
		let image_data = create_dummy_png(1000, 1000);
		{
			let mut zip = ZipWriter::new(&mut epub_data);

			// Add mimetype file
			zip.start_file(
				"mimetype",
				SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored),
			)?;
			zip.write_all(b"application/epub+zip")?;

			// Add a simple HTML file
			zip.start_file("OEBPS/content.xhtml", SimpleFileOptions::default())?;
			zip.write_all(br#"<html><body><img src="images/test.png"/></body></html>"#)?;

			// Add a dummy image
			zip.start_file("OEBPS/images/test.png", SimpleFileOptions::default())?;
			zip.write_all(&image_data)?;

			zip.finish()?;
		}

		// Create a temporary file to save the mock EPUB
		let mut temp_file = NamedTempFile::new()?;
		epub_data.seek(SeekFrom::Start(0))?;
		std::io::copy(&mut epub_data, &mut temp_file)?;

		// Create a temporary file for the output
		let output_file = NamedTempFile::new()?;

		// Create the Cli struct with our temp files
		let cli = Cli {
			input: temp_file.path().to_path_buf(),
			output: Some(output_file.path().to_path_buf()),
			jpeg_quality: 75,
			max_dimension: 500, // Set to a smaller value to ensure resizing
			hash_distance: 6,
			verbose: true,
			no_progress: false,
		};

		// Run the optimize function
		// Run the optimize function
		optimize(&cli)?;

		// Verify that the output file exists and is smaller than the input
		let input_size = temp_file.as_file().metadata()?.len();
		let output_size = output_file.as_file().metadata()?.len();

		assert!(output_size > 0, "Output file should not be empty");
		assert!(output_size < input_size, "Output file should be smaller than input");

		// Verify the content of the output EPUB
		let mut output_zip = ZipArchive::new(File::open(output_file.path())?)?;

		// Check if the mimetype file is present and correct
		let mut mimetype = String::new();
		output_zip.by_name("mimetype")?.read_to_string(&mut mimetype)?;
		assert_eq!(mimetype, "application/epub+zip");

		// Check if the HTML content was updated correctly
		let mut html_content = String::new();
		output_zip.by_name("OEBPS/content.xhtml")?.read_to_string(&mut html_content)?;
		assert!(html_content.contains("images/test.jpg"), "Image reference should be updated to JPG");

		// Check if the image was actually optimized
		let mut optimized_image = Vec::new();
		output_zip.by_name("OEBPS/images/test.jpg")?.read_to_end(&mut optimized_image)?;

		// Verify that the optimized image is smaller
		assert!(optimized_image.len() < image_data.len(), "Optimized image should be smaller");

		// Verify that the optimized image dimensions are correct
		let img = image::load_from_memory(&optimized_image)?;
		assert!(
			img.width() <= 500 && img.height() <= 500,
			"Image should be resized to 500x500 or smaller"
		);

		Ok(())
	}

	fn create_dummy_png(width: u32, height: u32) -> Vec<u8> {
		let mut img = image::RgbImage::new(width, height);
		for pixel in img.pixels_mut() {
			*pixel = image::Rgb([255, 255, 255]);
		}
		let mut buffer = Vec::new();
		img.write_to(&mut Cursor::new(&mut buffer), image::ImageFormat::Png).unwrap();
		buffer
	}

	#[test]
	fn test_comp_jpeg() -> Result<()> {
		let image = DynamicImage::new_rgb8(100, 100);
		let compressed = comp_jpeg(image, 75.0)?;
		assert!(!compressed.is_empty());
		Ok(())
	}

	#[test]
	fn test_needs_alpha() {
		let rgb_image = DynamicImage::new_rgb8(10, 10);
		assert!(!needs_alpha(&rgb_image));

		let mut rgba_image = DynamicImage::new_rgba8(10, 10);
		for pixel in rgba_image.as_mut_rgba8().unwrap().pixels_mut() {
			pixel[3] = 128; // Set alpha to 128 for all pixels
		}
		assert!(needs_alpha(&rgba_image));
	}

	#[test]
	fn test_crop_transparent_and_black() {
		let mut image = DynamicImage::new_rgba8(100, 100);
		// Draw a white rectangle in the middle
		for y in 25..75 {
			for x in 25..75 {
				image.draw_pixel(x, y, Rgba([255, 255, 255, 255]));
			}
		}
		let cropped = crop_transparent_and_black(image);
		assert!(cropped.width() < 100 && cropped.height() < 100);
	}

	#[test]
	fn test_calculate_image_hash() -> Result<()> {
		let image_data = create_dummy_png(100, 100);
		let hash = calculate_image_hash(&image_data)?;
		assert_eq!(hash.as_bytes().len(), 8);
		Ok(())
	}

	#[test]
	fn test_optimize_image() -> Result<()> {
		let image_data = create_dummy_png(100, 100);
		let (format, optimized, _) = optimize_image(&Default::default(), "test.png", &image_data, 50, 75.0)?;
		assert!(format == "jpg" || format == "png");
		assert!(optimized.len() < image_data.len());
		Ok(())
	}

	#[test]
	fn test_add_mimetype_file() -> Result<()> {
		let temp_dir = tempdir()?;
		let file_path = temp_dir.path().join("test.zip");
		let file = File::create(&file_path)?;
		let mut zip = ZipWriter::new(file);

		add_mimetype_file(&mut zip)?;
		zip.finish()?;

		let mut zip_reader = ZipArchive::new(File::open(&file_path)?)?;

		let mut mimetype_file = zip_reader.by_name("mimetype")?;
		assert_eq!(mimetype_file.compression(), zip::CompressionMethod::Stored);

		let mut content = String::new();
		mimetype_file.read_to_string(&mut content)?;
		assert_eq!(content, "application/epub+zip");

		Ok(())
	}

	#[test]
	fn test_collect_image_paths() -> Result<()> {
		let mut zip_data = Cursor::new(Vec::new());
		{
			let mut zip = ZipWriter::new(&mut zip_data);
			zip.start_file("image1.png", SimpleFileOptions::default())?;
			zip.write_all(b"fake png data")?;
			zip.start_file("image2.jpg", SimpleFileOptions::default())?;
			zip.write_all(b"fake jpg data")?;
			zip.start_file("not_an_image.txt", SimpleFileOptions::default())?;
			zip.write_all(b"not an image")?;
			zip.finish()?;
		}

		let mut zip = ZipArchive::new(Cursor::new(zip_data.into_inner()))?;
		let image_paths = collect_image_paths(&mut zip)?;

		assert_eq!(image_paths.len(), 2);
		assert!(image_paths.contains_key("image1.png"));
		assert!(image_paths.contains_key("image2.jpg"));
		assert!(!image_paths.contains_key("not_an_image.txt"));

		Ok(())
	}

	#[test]
	fn test_replace_image_references() -> Result<()> {
		// Create a dummy PNG image
		let image_data = create_dummy_png(100, 100);

		// Create an in-memory ZIP file with the image
		let mut zip_data = Cursor::new(Vec::new());
		{
			let mut zip = ZipWriter::new(&mut zip_data);
			zip.start_file("images/test.png", SimpleFileOptions::default())?;
			zip.write_all(&image_data)?;
			zip.finish()?;
		}
		zip_data.seek(SeekFrom::Start(0))?;

		let regex = &FILE_REFS_REGEX;
		let cli = Cli {
			input: PathBuf::new(),
			output: None,
			jpeg_quality: 75,
			max_dimension: 1440,
			hash_distance: 6,
			verbose: false,
			no_progress: true,
		};
		let immutable_state = ImmutableState {
			cli: &cli,
			image_paths: [("images/test.png".to_string(), "images/test.png".to_string())]
				.iter()
				.cloned()
				.collect(),
			progress_bar: None,
		};
		let mut mutable_state = MutableState {
			zip: ZipArchive::new(zip_data)?,
			outzip: ZipWriter::new(Cursor::new(Vec::new())),
			image_hashes: HashMap::new(),
			optimized_images: HashMap::new(),
			stats: Statistics::default(),
		};

		let content = r#"<img src="../images/test.png" alt="Test image">"#;
		let html_folder = Path::new("chapter1");

		let result = replace_image_references(regex, &immutable_state, &mut mutable_state, content, html_folder);

		// The result should now contain a reference to a JPG file (optimized version)
		assert!(result.contains("../images/test.jpg"), "Image reference should be updated to JPG");

		// Check if the optimized image was created
		assert!(
			mutable_state.optimized_images.contains_key("images/test.png"),
			"Optimized image should be created"
		);

		Ok(())
	}

	#[test]
	fn test_make_rel_path() {
		let current_folder = Path::new("OEBPS/chapter1");
		let target = Path::new("OEBPS/images/test.jpg");
		let rel_path = make_rel_path(current_folder, target);
		assert_eq!(rel_path, PathBuf::from("../images/test.jpg"));
	}

	#[test]
	fn test_join_path() {
		let path = Path::new("OEBPS/chapter1");
		let img = "../images/test.jpg";
		let joined = join_path(path, img);
		assert_eq!(joined, PathBuf::from("OEBPS/images/test.jpg"));
	}
}
