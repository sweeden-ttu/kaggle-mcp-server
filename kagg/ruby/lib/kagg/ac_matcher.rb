# frozen_string_literal: true

module Kagg
  # Aho-Corasick multi-pattern automaton for matching conference names,
  # venue keywords, and top ML Systems Engineering datasets.
  #
  # Pure-Ruby implementation: builds a trie with failure links, then
  # scans input text in O(n + m) time.
  module AcMatcher
    CONFERENCE_PATTERNS = [
      "AAAI", "AAAI Conference", "AAAI 2027",
      "International Conference on Learning Representations",
      "International Learning Representations",
      "ICLR", "ICLR 2027",
      "NeurIPS", "Neural Information Processing Systems",
      "ICML", "International Conference on Machine Learning",
      "CVPR", "Computer Vision and Pattern Recognition",
      "ACL", "Association for Computational Linguistics",
      "EMNLP", "Empirical Methods in Natural Language Processing",
      "KDD", "Knowledge Discovery and Data Mining",
      "SIGMOD", "VLDB", "IJCAI",
      "International Joint Conference on Artificial Intelligence",
      "MLSys", "Machine Learning and Systems",
    ].freeze

    DATASET_PATTERNS = [
      # Tabular / structured
      "Titanic", "House Prices", "Spaceship Titanic", "Store Sales",
      "Playground Series", "Porto Seguro", "IEEE-CIS Fraud Detection",
      "Santander Customer", "Home Credit Default Risk", "Elo Merchant",
      "Microsoft Malware Prediction", "Corporacion Favorita",
      "Instacart Market Basket", "Walmart Sales", "Rossmann Store Sales",
      "Allstate Claims", "Prudential Life Insurance", "Bosch Production Line",
      "Talking Data", "Avazu CTR", "Criteo Display Ads",
      # NLP
      "GLUE", "SuperGLUE", "SQuAD", "CoNLL", "SNLI", "MultiNLI",
      "SST-2", "MNLI", "Quora Question Pairs", "Natural Questions",
      "TriviaQA", "CommonCrawl", "C4 Dataset", "The Pile", "RedPajama",
      "MMLU", "HellaSwag", "WinoGrande", "ARC Challenge", "TruthfulQA",
      "HumanEval", "MBPP", "Jigsaw Toxic Comment", "Disaster Tweets",
      "Sentiment140",
      # Vision
      "ImageNet", "CIFAR-10", "CIFAR-100", "MNIST", "Fashion-MNIST",
      "COCO", "MS COCO", "Pascal VOC", "Open Images", "LSUN", "CelebA",
      "LFW", "Places365", "ADE20K", "Cityscapes", "KITTI",
      "Dogs vs Cats", "Plant Pathology", "Chest X-Ray", "Digit Recognizer",
      "Cassava Leaf Disease",
      # Audio / speech
      "LibriSpeech", "Common Voice", "AudioSet", "VoxCeleb", "RAVDESS",
      # Recommendation / ranking
      "MovieLens", "Netflix Prize", "Amazon Product Reviews",
      "Yelp Dataset", "Book-Crossing", "Steam Reviews",
      # Time series / forecasting
      "M5 Forecasting", "Web Traffic Forecasting", "COVID-19 Dataset",
      "Electricity Load", "ETTh1", "ETTm1",
      # Reinforcement learning
      "Atari 2600", "MuJoCo", "OpenAI Gym", "Procgen", "MineRL",
      # Systems / MLOps
      "MLPerf", "DAWNBench", "TPC-H", "TPC-DS", "BigBench", "DeepMind Lab",
      # Graph / relational
      "OGB", "Open Graph Benchmark", "Cora", "Citeseer", "PPI", "Reddit Dataset",
    ].freeze

    Match = Struct.new(:pattern, :category, :start, :end_pos, keyword_init: true)
    ScanResult = Struct.new(:matches, :category_counts, :unique_patterns, keyword_init: true)

    # ------------------------------------------------------------------
    # Trie node for Aho-Corasick
    # ------------------------------------------------------------------

    class TrieNode
      attr_accessor :children, :fail, :outputs

      def initialize
        @children = {}
        @fail = nil
        @outputs = []
      end
    end

    # ------------------------------------------------------------------
    # Automaton
    # ------------------------------------------------------------------

    class Automaton
      attr_reader :pattern_count

      def initialize
        @root = TrieNode.new
        @patterns = []
        @pattern_count = 0

        CONFERENCE_PATTERNS.each { |p| add_pattern(p, :conference) }
        DATASET_PATTERNS.each   { |p| add_pattern(p, :dataset) }

        build_failure_links
      end

      def scan(text)
        matches = []
        seen = Set.new
        category_counts = Hash.new(0)
        node = @root
        downcased = text.downcase

        downcased.each_char.with_index do |ch, i|
          node = follow_fail(node, ch)

          node.outputs.each do |pat_idx|
            pat, cat = @patterns[pat_idx]
            start_pos = i - pat.size + 1
            matches << Match.new(
              pattern: pat,
              category: cat,
              start: start_pos,
              end_pos: i + 1
            )
            category_counts[cat.to_s] += 1
            seen << pat
          end
        end

        ScanResult.new(
          matches: matches,
          category_counts: category_counts,
          unique_patterns: seen.to_a.sort
        )
      end

      def patterns_for(category)
        @patterns.select { |_, c| c == category }.map(&:first)
      end

      private

      def add_pattern(pattern, category)
        idx = @patterns.size
        @patterns << [pattern, category]
        @pattern_count += 1

        node = @root
        pattern.downcase.each_char do |ch|
          node.children[ch] ||= TrieNode.new
          node = node.children[ch]
        end
        node.outputs << idx
      end

      def build_failure_links
        queue = []
        @root.children.each_value do |child|
          child.fail = @root
          queue << child
        end

        until queue.empty?
          current = queue.shift
          current.children.each do |ch, child|
            queue << child
            fail_node = current.fail
            fail_node = fail_node.fail while fail_node && !fail_node.children.key?(ch)
            child.fail = fail_node ? fail_node.children[ch] : @root
            child.fail = @root if child.fail == child
            child.outputs += child.fail.outputs
          end
        end
      end

      def follow_fail(node, ch)
        while node && !node.children.key?(ch)
          node = node.fail
        end
        node ? node.children[ch] : @root
      end
    end
  end
end
