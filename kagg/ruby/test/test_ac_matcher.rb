# frozen_string_literal: true

require "minitest/autorun"
require "set"
require_relative "../lib/kagg"

class TestAcMatcher < Minitest::Test
  def setup
    @m = Kagg::AcMatcher::Automaton.new
  end

  def test_pattern_count
    assert @m.pattern_count > 100
  end

  def test_aaai_match
    result = @m.scan("Submitted to AAAI 2027 conference")
    pats = result.matches.map(&:pattern)
    assert_includes pats, "AAAI 2027"
    assert result.category_counts.key?("conference")
  end

  def test_iclr_full_name
    result = @m.scan("Published at the International Conference on Learning Representations")
    pats = result.matches.map(&:pattern)
    assert_includes pats, "International Conference on Learning Representations"
  end

  def test_international_learning_representations
    result = @m.scan("International Learning Representations venue")
    pats = result.matches.map(&:pattern)
    assert_includes pats, "International Learning Representations"
  end

  def test_dataset_imagenet_cifar
    result = @m.scan("We benchmark on ImageNet and CIFAR-10")
    ds = result.matches.select { |h| h.category == :dataset }.map(&:pattern)
    assert_includes ds, "ImageNet"
    assert_includes ds, "CIFAR-10"
  end

  def test_dataset_titanic
    result = @m.scan("Kaggle Titanic competition is a classic")
    pats = result.matches.map(&:pattern)
    assert_includes pats, "Titanic"
  end

  def test_case_insensitive
    result = @m.scan("we evaluated on mnist and cifar-100")
    pats = result.matches.map(&:pattern)
    assert_includes pats, "MNIST"
    assert_includes pats, "CIFAR-100"
  end

  def test_mixed
    text = "Our AAAI 2027 paper evaluates on SQuAD, MMLU, and ImageNet using MLPerf benchmarks"
    result = @m.scan(text)
    assert(result.category_counts.fetch("conference", 0) > 0)
    assert(result.category_counts.fetch("dataset", 0) > 0)
    assert result.unique_patterns.size >= 4
  end

  def test_no_false_positives
    result = @m.scan("The quick brown fox jumps over the lazy dog")
    assert_empty result.matches
  end

  def test_patterns_for_category
    confs = @m.patterns_for(:conference)
    assert_includes confs, "AAAI"
    assert_includes confs, "ICLR"

    datasets = @m.patterns_for(:dataset)
    assert_includes datasets, "Titanic"
    assert_includes datasets, "ImageNet"
  end
end
