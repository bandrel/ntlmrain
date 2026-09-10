mod auth;
mod backend;
mod config;
mod http;
mod queue;
mod worker;

use config::Config;

fn main() {
    let config = Config::parse();
    if let Err(error) = config.validate() {
        eprintln!("ntlmrain-server: {error}");
        std::process::exit(1);
    }
    let _ = config;
    eprintln!("config OK, server not yet implemented");
}
