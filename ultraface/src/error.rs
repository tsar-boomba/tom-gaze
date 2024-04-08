use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
	#[error(transparent)]
	Ort(#[from] ort::Error)
}
