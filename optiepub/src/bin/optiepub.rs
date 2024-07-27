use eyre::Result;

fn main() -> Result<()> {
	optiepub::optimize(&argh::from_env())
}
