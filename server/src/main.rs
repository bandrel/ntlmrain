mod backend;
mod config;

use clap::Parser;
use config::Config;

fn main() {
    let config = Config::parse();
    if let Err(error) = config.validate() {
        eprintln!("ntlmrain-server: {error}");
        std::process::exit(1);
    }
    println!("config OK, server not yet implemented");
    println!("{config:#?}");
}
