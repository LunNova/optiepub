fn main() -> eyre::Result<()> {
	optiepub::optimize(&argh::from_env())
}
