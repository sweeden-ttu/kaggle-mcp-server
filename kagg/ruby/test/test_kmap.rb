# frozen_string_literal: true

require "minitest/autorun"
require "set"
require_relative "../lib/kagg"

class TestGrayCode < Minitest::Test
  def test_base_cases
    assert_equal [0], Kagg::KMap.gray_code(0)
    assert_equal [0, 1], Kagg::KMap.gray_code(1)
    assert_equal [0, 1, 3, 2], Kagg::KMap.gray_code(2)
    assert_equal [0, 1, 3, 2, 6, 7, 5, 4], Kagg::KMap.gray_code(3)
  end

  def test_adjacency
    (1..5).each do |n|
      codes = Kagg::KMap.gray_code(n)
      assert_equal 1 << n, codes.size
      codes.each_cons(2) do |a, b|
        diff = a ^ b
        assert (diff & (diff - 1)).zero?, "Adjacent codes must differ in exactly one bit"
      end
    end
  end
end

class TestKMap < Minitest::Test
  def test_2var_kmap
    km = Kagg::KMap::Map.new([0, 3], 2)
    assert_equal 2, km.rows
    assert_equal 2, km.cols
    assert_equal [0, 3], km.minterms
  end

  def test_simplify_single_minterm
    km = Kagg::KMap::Map.new([5], 3, variables: %w[A B C])
    result = Kagg::KMap.simplify(km)
    refute_empty result.sop
    refute_equal "0", result.sop
  end

  def test_simplify_empty_minterms_returns_zero
    km = Kagg::KMap::Map.new([], 2)
    result = Kagg::KMap.simplify(km)
    assert_equal "0", result.sop
    assert_empty result.prime_implicants
  end
end

class TestEquivalence < Minitest::Test
  def test_demorgan
    # !(A && B) == (!A || !B)
    lhs = Kagg::KMap::NotExpr.new(
      Kagg::KMap::AndExpr.new(Kagg::KMap::Var.new("A"), Kagg::KMap::Var.new("B"))
    )
    rhs = Kagg::KMap::OrExpr.new(
      Kagg::KMap::NotExpr.new(Kagg::KMap::Var.new("A")),
      Kagg::KMap::NotExpr.new(Kagg::KMap::Var.new("B"))
    )
    result = Kagg::KMap.equivalence_table(lhs, rhs)
    assert result.is_equivalent
    assert_nil result.counterexample
  end

  def test_tautology
    a = Kagg::KMap::OrExpr.new(
      Kagg::KMap::Var.new("x"),
      Kagg::KMap::NotExpr.new(Kagg::KMap::Var.new("x"))
    )
    b = Kagg::KMap::Const.new(true)
    result = Kagg::KMap.equivalence_table(a, b)
    assert result.is_equivalent
  end

  def test_not_equivalent
    a = Kagg::KMap::Var.new("x")
    b = Kagg::KMap::NotExpr.new(Kagg::KMap::Var.new("x"))
    result = Kagg::KMap.equivalence_table(a, b)
    refute result.is_equivalent
    refute_nil result.counterexample
  end

  def test_implies_not_equivalent_to_iff
    imp = Kagg::KMap::ImpliesExpr.new(
      Kagg::KMap::Var.new("A"), Kagg::KMap::Var.new("B")
    )
    iff = Kagg::KMap::IffExpr.new(
      Kagg::KMap::Var.new("A"), Kagg::KMap::Var.new("B")
    )
    result = Kagg::KMap.equivalence_table(imp, iff)
    refute result.is_equivalent, "Implies and Iff must not be equivalent"
    refute_nil result.counterexample
  end

  def test_xor_not_equivalent_to_or
    xor = Kagg::KMap::XorExpr.new(
      Kagg::KMap::Var.new("A"), Kagg::KMap::Var.new("B")
    )
    or_expr = Kagg::KMap::OrExpr.new(
      Kagg::KMap::Var.new("A"), Kagg::KMap::Var.new("B")
    )
    result = Kagg::KMap.equivalence_table(xor, or_expr)
    refute result.is_equivalent
  end
end

class TestBoolExprBase < Minitest::Test
  def test_base_evaluate_raises_not_implemented
    base = Kagg::KMap::BoolExpr.new
    assert_raises(NotImplementedError) { base.evaluate({}) }
  end
end

class TestParser < Minitest::Test
  def test_parse_and
    expr = Kagg::KMap.parse("And(A, B)")
    assert_instance_of Kagg::KMap::AndExpr, expr
    assert_equal %w[A B], expr.variables
  end

  def test_parse_nested
    expr = Kagg::KMap.parse("Or(Not(A), Iff(B, C))")
    assert_instance_of Kagg::KMap::OrExpr, expr
    assert_equal %w[A B C], expr.variables
  end

  def test_parse_bad_token_raises
    assert_raises(RuntimeError) { Kagg::KMap.parse("BadFunc(A)") }
  end
end

class TestKMapFromExpr < Minitest::Test
  def test_and_expr_kmap
    expr = Kagg::KMap::AndExpr.new(
      Kagg::KMap::Var.new("A"), Kagg::KMap::Var.new("B")
    )
    km = Kagg::KMap.kmap_from_expr(expr)
    assert_equal 2, km.num_vars
    assert_equal [3], km.minterms
  end
end
