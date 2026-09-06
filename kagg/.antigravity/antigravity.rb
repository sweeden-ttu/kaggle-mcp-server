# frozen_string_literal: true

# .antigravity/antigravity.rb — Ruby bootstrap entry point
#
# Detects the kagg gem, loads it, and provides a one-liner REPL
# for K-map simplification and equivalence checking.

$LOAD_PATH.unshift(File.expand_path("../ruby/lib", __dir__))
require "kagg"

module Antigravity
  def self.launch(args = ARGV)
    case args.first
    when "kmap"
      vars = extract_opt(args, "--vars")&.to_i || 4
      mt   = extract_opt(args, "--minterms")&.split(",")&.map(&:to_i) || []
      dc   = extract_opt(args, "--dont-cares")&.split(",")&.map(&:to_i)

      km = Kagg::KMap::Map.new(mt, vars, dont_cares: dc)
      puts km.display
      result = Kagg::KMap.simplify(km)
      puts "Simplified: #{result.sop}"

    when "equiv"
      a_str = extract_opt(args, "--a") || abort("--a required")
      b_str = extract_opt(args, "--b") || abort("--b required")
      ea = Kagg::KMap.parse(a_str)
      eb = Kagg::KMap.parse(b_str)
      puts Kagg::KMap.equivalence_table(ea, eb).display

    when "truth-table"
      expr_str = extract_opt(args, "--expr") || abort("--expr required")
      expr = Kagg::KMap.parse(expr_str)
      tt = Kagg::KMap.truth_table(expr)
      vars = expr.variables
      puts vars.join(" | ") + " | f"
      puts "-" * (vars.size * 4 + 4)
      tt.each do |assignment, val|
        row = vars.map { |v| assignment[v] ? "1" : "0" }.join(" | ")
        puts "#{row} | #{val ? '1' : '0'}"
      end

    when "stats"
      db = Kagg::Database.new
      require "json"
      puts JSON.pretty_generate(db.stats)

    else
      warn "Usage: antigravity.rb {kmap|equiv|truth-table|stats} [options]"
      exit 1
    end
  end

  def self.extract_opt(args, flag)
    idx = args.index(flag)
    idx ? args[idx + 1] : nil
  end
end

Antigravity.launch if __FILE__ == $PROGRAM_NAME
