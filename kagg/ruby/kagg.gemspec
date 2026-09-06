# frozen_string_literal: true

Gem::Specification.new do |s|
  s.name        = "kagg"
  s.version     = "0.1.0"
  s.summary     = "MLSysEng MoE – Ruby edition"
  s.description = "Recursively enumerable K-map, logical equivalence predicate tables, and MoE expert system"
  s.authors     = ["sweeden-ttu"]
  s.license     = "MIT"
  s.files       = Dir["lib/**/*.rb"]
  s.require_paths = ["lib"]
  s.required_ruby_version = ">= 3.1"

  s.add_dependency "sqlite3", "~> 1.7"
end
