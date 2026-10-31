use std::path::Path;

use orp::params::RuntimeParameters;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

use crate::util::result::Result;

pub(crate) fn open_session(path: impl AsRef<Path>, params: &RuntimeParameters) -> Result<Session> {
    Ok(Session::builder()?
        .with_intra_threads(params.threads())?
        .with_execution_providers(params.execution_providers().to_vec())?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(path)?)
}
