;; gorilla-repl.fileformat = 1

;; **
;;; # Testing FOPPL compilation to graph
;; **

;; @@
(ns foppl-test
  (:require [gorilla-plot.core :as plot]
            [foppl.desugar :refer :all]
            [foppl.gibbs :as gibbs]
            [foppl.core :as foppl :refer [foppl-query print-graph]] :reload)
  (:use [anglican runtime]))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; **
;;; ## Basic idea:
;;; 
;;; Calling `foppl-query` with some pseudo-Anglican code inside will return a tuple with two entries; the first is a graph, the second is an expression for the return value.
;;; 
;;; `print-graph` is a useful helper function.
;;; 
;;; Here are a few very simple example programs.
;; **

;; @@
(let [[G E]
      (foppl-query
        (let [data (vector 1 2 3)
              a (vector 2)]
          (vector (first (rest (rest data))) a)))]
  (print-graph G)
  (:body E))
;; @@
;; ->
;;; Vertices V: #{}
;;; 
;;; Arcs A: #{}
;;; 
;;; Conditional densities P:
;;; 
;;; 
;;; Observed values O:
;;; 
;;; 
;; <-
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-long'>3</span>","value":"3"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-long'>2</span>","value":"2"}],"value":"[2]"}],"value":"[3 [2]]"}
;; <=

;; @@
(let [[G E]
      (foppl-query
        (let [data (vector 1 2 (sample (normal 1 1)))
              a (conj [] (sample (normal 0 2)))
              b (conj a (sample (normal 0 3)))]
          (observe (normal (second b) 4)
                   (first (rest data)))
          b))]
  (print-graph G)
  (:body E))
;; @@
;; ->
;;; Vertices V: #{y27286 x27275 x27270 x27265}
;;; 
;;; Arcs A: #{[x27275 y27286]}
;;; 
;;; Conditional densities P:
;;; x27265 -&gt; (fn [] (normal 1 1))
;;; x27270 -&gt; (fn [] (normal 0 2))
;;; x27275 -&gt; (fn [] (normal 0 3))
;;; y27286 -&gt; (fn [x27275] (normal x27275 4))
;;; 
;;; Observed values O:
;;; y27286 -&gt; 2
;;; 
;; <-
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x27270</span>","value":"x27270"},{"type":"html","content":"<span class='clj-symbol'>x27275</span>","value":"x27275"}],"value":"[x27270 x27275]"}
;; <=

;; @@
(let [[G E]
      (foppl-query
        (let [data (vector 1 2 (sample (normal 1 1)))
              a (conj [] (/ (sample (normal 0 2)) 2))
              b (conj a (sample (normal 0 3)))]
          (observe (normal (second b) 4)
                   (first (rest data)))
          b))]
  (print-graph G)
  (:body E))
;; @@
;; ->
;;; Vertices V: #{x27331 x27326 y27347 x27336}
;;; 
;;; Arcs A: #{[x27336 y27347]}
;;; 
;;; Conditional densities P:
;;; x27326 -&gt; (fn [] (normal 1 1))
;;; x27331 -&gt; (fn [] (normal 0 2))
;;; x27336 -&gt; (fn [] (normal 0 3))
;;; y27347 -&gt; (fn [x27336] (normal x27336 4))
;;; 
;;; Observed values O:
;;; y27347 -&gt; 2
;;; 
;; <-
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-list'>(</span>","close":"<span class='clj-list'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>/</span>","value":"/"},{"type":"html","content":"<span class='clj-symbol'>x27331</span>","value":"x27331"},{"type":"html","content":"<span class='clj-long'>2</span>","value":"2"}],"value":"(/ x27331 2)"},{"type":"html","content":"<span class='clj-symbol'>x27336</span>","value":"x27336"}],"value":"[(/ x27331 2) x27336]"}
;; <=

;; @@
(let [[G E]
      (foppl-query
        (let [x (sample (normal 0 1))]
          x))]
  (print-graph G)
  (:body E))
;; @@
;; ->
;;; Vertices V: #{x27386}
;;; 
;;; Arcs A: #{}
;;; 
;;; Conditional densities P:
;;; x27386 -&gt; (fn [] (normal 0 1))
;;; 
;;; Observed values O:
;;; 
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-symbol'>x27386</span>","value":"x27386"}
;; <=

;; @@
(let [[G E]
      (foppl-query
        (let [x (sample (normal 0 1))]
          (sample (normal x 1))))]
  (print-graph G)
  (:body E))
;; @@
;; ->
;;; Vertices V: #{x27408 x27407}
;;; 
;;; Arcs A: #{[x27407 x27408]}
;;; 
;;; Conditional densities P:
;;; x27407 -&gt; (fn [] (normal 0 1))
;;; x27408 -&gt; (fn [x27407] (normal x27407 1))
;;; 
;;; Observed values O:
;;; 
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-symbol'>x27408</span>","value":"x27408"}
;; <=

;; **
;;; ## Beta-flip (single observe).
;;; 
;;; The actual core language only permits a single binding, and a single body statement, in each let block.
;;; 
;;; This example demonstrates the `let` desugaring.
;; **

;; @@
(let [[G E]
      (foppl-query
        (let [p (sample (beta 1 1))
              x (sample (beta (exp p) 1))
              d (bernoulli (* x p))]
          (observe d 1)
          p))]
  (print-graph G)
  (:body E))
;; @@
;; ->
;;; Vertices V: #{x27436 x27437 y27438}
;;; 
;;; Arcs A: #{[x27436 x27437] [x27436 y27438] [x27437 y27438]}
;;; 
;;; Conditional densities P:
;;; x27436 -&gt; (fn [] (beta 1 1))
;;; x27437 -&gt; (fn [x27436] (beta (exp x27436) 1))
;;; y27438 -&gt; (fn [x27437 x27436] (bernoulli (* x27437 x27436)))
;;; 
;;; Observed values O:
;;; y27438 -&gt; 1
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-symbol'>x27436</span>","value":"x27436"}
;; <=

;; **
;;; ## Linear regression. 
;;; 
;;; Uses `loop` and user-defined functions.
;;; 
;;; User-defined functions must be defined prior to the main body expression of the program. They can only call functions defined above themselves.
;;; 
;;; The `loop` construct is basically a `reduce`, but with a statically determined number of loop iterations. The function signature is 
;;; 
;;; `(loop num-times initial-value function & [args])`
;;; 
;;; which calls the function `function` a total of `num-times`. The function itself should look like
;;; 
;;; `(defn function [loop-index value & [args]] ...)`
;;; 
;;; where `loop-index` is an integer, and `value` is either the initial value, or the return value of the previous loop iteration.
;;; 
;;; Hopefully the linear regression example makes this construct clear:
;; **

;; @@
(def linear-regression
  (foppl-query
    (defn observe-data [_ data slope bias]
      (let [xn (first data)
            yn (second data)
            zn (+ (* slope xn) bias)]
        (observe (normal zn 1.0) yn)
        (rest (rest data))))

    (let [slope (sample (normal 0.0 10.0))
          bias  (sample (normal 0.0 10.0))
          data (vector 
                 1.0 2.1 2.0 3.9 3.0 5.3
                 4.0 7.7 5.0 10.2 6.0 12.9)]
      (loop 6 data observe-data slope bias)
      (vector slope bias))))

(let [[G E] linear-regression]
  (print-graph G)
  (:body E))
;; @@
;; ->
;;; Vertices V: #{y27534 y27559 x27481 y27509 x27478 y27488 y27609 y27584}
;;; 
;;; Arcs A: #{[x27481 y27534] [x27478 y27609] [x27481 y27509] [x27478 y27488] [x27481 y27584] [x27478 y27509] [x27478 y27584] [x27481 y27559] [x27478 y27559] [x27478 y27534] [x27481 y27609] [x27481 y27488]}
;;; 
;;; Conditional densities P:
;;; x27478 -&gt; (fn [] (normal 0.0 10.0))
;;; x27481 -&gt; (fn [] (normal 0.0 10.0))
;;; y27488 -&gt; (fn [x27481 x27478] (normal (+ (* x27478 1.0) x27481) 1.0))
;;; y27509 -&gt; (fn [x27481 x27478] (normal (+ (* x27478 2.0) x27481) 1.0))
;;; y27534 -&gt; (fn [x27481 x27478] (normal (+ (* x27478 3.0) x27481) 1.0))
;;; y27559 -&gt; (fn [x27481 x27478] (normal (+ (* x27478 4.0) x27481) 1.0))
;;; y27584 -&gt; (fn [x27481 x27478] (normal (+ (* x27478 5.0) x27481) 1.0))
;;; y27609 -&gt; (fn [x27481 x27478] (normal (+ (* x27478 6.0) x27481) 1.0))
;;; 
;;; Observed values O:
;;; y27488 -&gt; 2.1
;;; y27509 -&gt; 3.9
;;; y27534 -&gt; 5.3
;;; y27559 -&gt; 7.7
;;; y27584 -&gt; 10.2
;;; y27609 -&gt; 12.9
;;; 
;; <-
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x27478</span>","value":"x27478"},{"type":"html","content":"<span class='clj-symbol'>x27481</span>","value":"x27481"}],"value":"[x27478 x27481]"}
;; <=

;; @@
(foppl/draw-from-prior linear-regression)
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-record'>#foppl.core.Trace{</span>","close":"<span class='clj-record'>}</span>","separator":" ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:rdb</span>","value":":rdb"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x27478</span>","value":"x27478"},{"type":"html","content":"<span class='clj-double'>-0.5746536705797409</span>","value":"-0.5746536705797409"}],"value":"[x27478 -0.5746536705797409]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x27481</span>","value":"x27481"},{"type":"html","content":"<span class='clj-double'>-5.216129348584157</span>","value":"-5.216129348584157"}],"value":"[x27481 -5.216129348584157]"}],"value":"{x27478 -0.5746536705797409, x27481 -5.216129348584157}"}],"value":"[:rdb {x27478 -0.5746536705797409, x27481 -5.216129348584157}]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:ordering</span>","value":":ordering"},{"type":"list-like","open":"<span class='clj-list'>(</span>","close":"<span class='clj-list'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x27478</span>","value":"x27478"},{"type":"html","content":"<span class='clj-symbol'>x27481</span>","value":"x27481"},{"type":"html","content":"<span class='clj-symbol'>y27509</span>","value":"y27509"},{"type":"html","content":"<span class='clj-symbol'>y27488</span>","value":"y27488"},{"type":"html","content":"<span class='clj-symbol'>y27609</span>","value":"y27609"},{"type":"html","content":"<span class='clj-symbol'>y27584</span>","value":"y27584"},{"type":"html","content":"<span class='clj-symbol'>y27559</span>","value":"y27559"},{"type":"html","content":"<span class='clj-symbol'>y27534</span>","value":"y27534"}],"value":"(x27478 x27481 y27509 y27488 y27609 y27584 y27559 y27534)"}],"value":"[:ordering (x27478 x27481 y27509 y27488 y27609 y27584 y27559 y27534)]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:logprob</span>","value":":logprob"},{"type":"html","content":"<span class='clj-double'>-686.3255070634393</span>","value":"-686.3255070634393"}],"value":"[:logprob -686.3255070634393]"}],"value":"#foppl.core.Trace{:rdb {x27478 -0.5746536705797409, x27481 -5.216129348584157}, :ordering (x27478 x27481 y27509 y27488 y27609 y27584 y27559 y27534), :logprob -686.3255070634393}"}
;; <=

;; **
;;; ## Hidden Markov Model.
;; **

;; @@
(let [[G E]
      (foppl-query
        (defn data [n]
          (let [points (vector 0.9 0.8 0.7 0.0 -0.025
                               5.0 2.0 0.1 0.0 0.13 
                               0.45 6.0 0.2 0.3 -1.0 -1.0)]
            (get points n)))

        ;; Define the init, transition, and observation distributions
        (defn get-init-params []
          (vector (/ 1. 3.) (/ 1. 3.) (/ 1. 3.)))

        (defn get-trans-params [k]
          (nth (vector (vector 0.1  0.5  0.4 )
                       (vector 0.2  0.2  0.6 )
                       (vector 0.15 0.15 0.7 )) k))

        (defn get-obs-dist [k]
          (nth (vector (normal -1 1)
                       (normal  1 1)
                       (normal  0 1)) k))

        ;; Function to step through HMM and sample latent state
        (defn hmm-step [n states]
          (let [next-state (sample (discrete (get-trans-params (last states))))]
            (observe (get-obs-dist next-state) (data n))
            (conj states next-state)))

        ;; Loop through the data
        (let [init-state (sample (discrete (get-init-params)))]
          (loop 16 (vector init-state) hmm-step)))]
  (print-graph G)
  (println "\nNumber of arcs:" (count (second G)))
  (:body E))
;; @@
;; ->
;;; Vertices V: #{x27983 x27815 y27968 x27787 x28151 x28123 x27843 y27744 x27899 x28095 x27955 x28067 y28080 x28039 x28011 y28164 y27996 y28052 y27912 y27800 x27927 x27731 x27759 y27772 y28108 y27940 y28136 x27871 x27716 y27856 y28024 y27828 y27884}
;;; 
;;; Arcs A: #{[x27983 x28011] [x27843 y27856] [x27955 y27968] [x28011 y28024] [x27759 y27772] [x28095 x28123] [x28011 x28039] [x27899 y27912] [x27787 x27815] [x27815 y27828] [x27815 x27843] [x27955 x27983] [x28151 y28164] [x27983 y27996] [x27927 y27940] [x27843 x27871] [x27787 y27800] [x28039 x28067] [x28123 y28136] [x27899 x27927] [x28123 x28151] [x28095 y28108] [x28067 x28095] [x27871 y27884] [x27716 x27731] [x27871 x27899] [x27927 x27955] [x27731 y27744] [x27731 x27759] [x28067 y28080] [x27759 x27787] [x28039 y28052]}
;;; 
;;; Conditional densities P:
;;; x27983 -&gt; (fn [x27955] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x27955)))
;;; x27815 -&gt; (fn [x27787] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x27787)))
;;; y27968 -&gt; (fn [x27955] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x27955))
;;; x27787 -&gt; (fn [x27759] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x27759)))
;;; x28151 -&gt; (fn [x28123] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x28123)))
;;; x28123 -&gt; (fn [x28095] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x28095)))
;;; x27843 -&gt; (fn [x27815] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x27815)))
;;; y27744 -&gt; (fn [x27731] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x27731))
;;; x27899 -&gt; (fn [x27871] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x27871)))
;;; x28095 -&gt; (fn [x28067] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x28067)))
;;; x27955 -&gt; (fn [x27927] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x27927)))
;;; x28067 -&gt; (fn [x28039] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x28039)))
;;; y28080 -&gt; (fn [x28067] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x28067))
;;; x28039 -&gt; (fn [x28011] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x28011)))
;;; x28011 -&gt; (fn [x27983] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x27983)))
;;; y28164 -&gt; (fn [x28151] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x28151))
;;; y27996 -&gt; (fn [x27983] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x27983))
;;; y28052 -&gt; (fn [x28039] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x28039))
;;; y27912 -&gt; (fn [x27899] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x27899))
;;; y27800 -&gt; (fn [x27787] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x27787))
;;; x27927 -&gt; (fn [x27899] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x27899)))
;;; x27731 -&gt; (fn [x27716] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x27716)))
;;; x27759 -&gt; (fn [x27731] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x27731)))
;;; y27772 -&gt; (fn [x27759] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x27759))
;;; y28108 -&gt; (fn [x28095] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x28095))
;;; y27940 -&gt; (fn [x27927] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x27927))
;;; y28136 -&gt; (fn [x28123] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x28123))
;;; x27871 -&gt; (fn [x27843] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x27843)))
;;; x27716 -&gt; (fn [] (discrete [0.3333333333333333 0.3333333333333333 0.3333333333333333]))
;;; y27856 -&gt; (fn [x27843] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x27843))
;;; y28024 -&gt; (fn [x28011] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x28011))
;;; y27828 -&gt; (fn [x27815] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x27815))
;;; y27884 -&gt; (fn [x27871] (nth (vector (normal -1 1) (normal 1 1) (normal 0 1)) x27871))
;;; 
;;; Observed values O:
;;; y27968 -&gt; 0.0
;;; y27744 -&gt; 0.9
;;; y28080 -&gt; 0.2
;;; y28164 -&gt; -1.0
;;; y27996 -&gt; 0.13
;;; y28052 -&gt; 6.0
;;; y27912 -&gt; 2.0
;;; y27800 -&gt; 0.7
;;; y27772 -&gt; 0.8
;;; y28108 -&gt; 0.3
;;; y27940 -&gt; 0.1
;;; y28136 -&gt; -1.0
;;; y27856 -&gt; -0.025
;;; y28024 -&gt; 0.45
;;; y27828 -&gt; 0.0
;;; y27884 -&gt; 5.0
;;; 
;;; Number of arcs: 32
;;; 
;; <-
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x27716</span>","value":"x27716"},{"type":"html","content":"<span class='clj-symbol'>x27731</span>","value":"x27731"},{"type":"html","content":"<span class='clj-symbol'>x27759</span>","value":"x27759"},{"type":"html","content":"<span class='clj-symbol'>x27787</span>","value":"x27787"},{"type":"html","content":"<span class='clj-symbol'>x27815</span>","value":"x27815"},{"type":"html","content":"<span class='clj-symbol'>x27843</span>","value":"x27843"},{"type":"html","content":"<span class='clj-symbol'>x27871</span>","value":"x27871"},{"type":"html","content":"<span class='clj-symbol'>x27899</span>","value":"x27899"},{"type":"html","content":"<span class='clj-symbol'>x27927</span>","value":"x27927"},{"type":"html","content":"<span class='clj-symbol'>x27955</span>","value":"x27955"},{"type":"html","content":"<span class='clj-symbol'>x27983</span>","value":"x27983"},{"type":"html","content":"<span class='clj-symbol'>x28011</span>","value":"x28011"},{"type":"html","content":"<span class='clj-symbol'>x28039</span>","value":"x28039"},{"type":"html","content":"<span class='clj-symbol'>x28067</span>","value":"x28067"},{"type":"html","content":"<span class='clj-symbol'>x28095</span>","value":"x28095"},{"type":"html","content":"<span class='clj-symbol'>x28123</span>","value":"x28123"},{"type":"html","content":"<span class='clj-symbol'>x28151</span>","value":"x28151"}],"value":"[x27716 x27731 x27759 x27787 x27815 x27843 x27871 x27899 x27927 x27955 x27983 x28011 x28039 x28067 x28095 x28123 x28151]"}
;; <=

;; **
;;; ## Simple example of challenges with primitive procedures
;;; 
;;; This works now!
;; **

;; @@
(let [[G E]
      (foppl-query
        (let [a (vector (sample (normal 0 1)))
              b (conj a (sample (normal 0 1)))]
          (observe (normal (second b) 1) 1)))]
  (print-graph G)
  (:body E))
;; @@
;; ->
;;; Vertices V: #{y28389 x28379 x28384}
;;; 
;;; Arcs A: #{[x28384 y28389]}
;;; 
;;; Conditional densities P:
;;; x28379 -&gt; (fn [] (normal 0 1))
;;; x28384 -&gt; (fn [] (normal 0 1))
;;; y28389 -&gt; (fn [x28384] (normal x28384 1))
;;; 
;;; Observed values O:
;;; y28389 -&gt; 1
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-long'>1</span>","value":"1"}
;; <=

;; @@
(let [[G E]
      (foppl-query
        (let [a (vector (sample (normal 0 1)))
              b (conj a (sample (normal 0 1)))]
          (observe (normal (sum b) 1) 1)))]
  (print-graph G)
  (:body E))
;; @@
;; ->
;;; Vertices V: #{y28430 x28422 x28427}
;;; 
;;; Arcs A: #{[x28422 y28430] [x28427 y28430]}
;;; 
;;; Conditional densities P:
;;; x28422 -&gt; (fn [] (normal 0 1))
;;; x28427 -&gt; (fn [] (normal 0 1))
;;; y28430 -&gt; (fn [x28427 x28422] (normal (sum [x28422 x28427]) 1))
;;; 
;;; Observed values O:
;;; y28430 -&gt; 1
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-long'>1</span>","value":"1"}
;; <=

;; @@
(let [[G E]
      (foppl-query
        (defn first-arg [arg1 arg2] arg1)
        
        (defn second-arg [arg1 arg2] arg2)
    
        (let [a (sample (normal 0 1))
              b (sample (normal 0 1))]
          (observe (normal (first-arg a b) 1) 1)))]
  (print-graph G)
  (:body E))
;; @@
;; ->
;;; Vertices V: #{y28467 x28466 x28463}
;;; 
;;; Arcs A: #{[x28463 y28467]}
;;; 
;;; Conditional densities P:
;;; x28463 -&gt; (fn [] (normal 0 1))
;;; x28466 -&gt; (fn [] (normal 0 1))
;;; y28467 -&gt; (fn [x28463] (normal x28463 1))
;;; 
;;; Observed values O:
;;; y28467 -&gt; 1
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-long'>1</span>","value":"1"}
;; <=

;; **
;;; ## Gaussian mixture
;; **

;; @@
(let [[G E]
      (foppl-query
        (defn sample-likelihoods [_ likes]
          (let [precision (sample (gamma 1.0 1.0))
                mean (sample (normal 0.0 precision))
                sigma (/ (sqrt precision))]
            (conj likes
                  (normal mean sigma))))

        (defn sample-components [_ zs prior]
          (let [z (sample prior)]
            (conj zs z)))

        (defn observe-data [n _ ys zs likes]
          (let [y (nth ys n)
                z (nth zs n)]
            (observe (nth likes z) y)
            nil))

        (let [ys (vector 1.1 2.1 2.0 1.9 0.0 -0.1 -0.05)
              z-prior (discrete 
                        (sample (dirichlet (vector 1.0 1.0 1.0))))
              zs (loop 7 (vector) sample-components z-prior)
              likes (loop 3 (vector) sample-likelihoods)]
          (loop 7 nil observe-data ys zs likes)
          zs))]
  (print-graph G)
  (println "\nNumber of arcs:" (count (second G)))
  (:body E))
;; @@
;; ->
;;; Vertices V: #{y28580 x28564 x28544 y28615 y28594 x28526 y28573 x28523 x28541 x28538 x28551 x28558 x28552 y28608 x28563 x28557 x28532 y28601 x28535 y28587 x28529}
;;; 
;;; Arcs A: #{[x28564 y28580] [x28563 y28580] [x28563 y28573] [x28551 y28594] [x28558 y28594] [x28529 y28580] [x28557 y28608] [x28563 y28594] [x28551 y28587] [x28551 y28573] [x28564 y28615] [x28551 y28580] [x28523 x28538] [x28563 x28564] [x28523 x28529] [x28551 y28601] [x28552 y28601] [x28557 y28580] [x28532 y28587] [x28563 y28587] [x28523 x28526] [x28558 y28580] [x28558 y28608] [x28523 x28541] [x28557 y28615] [x28552 y28615] [x28538 y28601] [x28552 y28580] [x28557 y28573] [x28552 y28594] [x28523 x28535] [x28558 y28573] [x28552 y28608] [x28535 y28594] [x28551 y28615] [x28563 y28601] [x28558 y28601] [x28558 y28615] [x28551 y28608] [x28563 y28615] [x28541 y28608] [x28564 y28601] [x28557 y28594] [x28557 y28601] [x28564 y28573] [x28558 y28587] [x28552 y28573] [x28551 x28552] [x28557 x28558] [x28526 y28573] [x28557 y28587] [x28523 x28532] [x28564 y28594] [x28564 y28587] [x28552 y28587] [x28563 y28608] [x28544 y28615] [x28523 x28544] [x28564 y28608]}
;;; 
;;; Conditional densities P:
;;; y28580 -&gt; (fn [x28558 x28551 x28563 x28564 x28529 x28557 x28552] (nth [(normal x28552 (/ (sqrt x28551))) (normal x28558 (/ (sqrt x28557))) (normal x28564 (/ (sqrt x28563)))] x28529))
;;; x28564 -&gt; (fn [x28563] (normal 0.0 x28563))
;;; x28544 -&gt; (fn [x28523] (discrete x28523))
;;; y28615 -&gt; (fn [x28558 x28551 x28544 x28563 x28564 x28557 x28552] (nth [(normal x28552 (/ (sqrt x28551))) (normal x28558 (/ (sqrt x28557))) (normal x28564 (/ (sqrt x28563)))] x28544))
;;; y28594 -&gt; (fn [x28535 x28558 x28551 x28563 x28564 x28557 x28552] (nth [(normal x28552 (/ (sqrt x28551))) (normal x28558 (/ (sqrt x28557))) (normal x28564 (/ (sqrt x28563)))] x28535))
;;; x28526 -&gt; (fn [x28523] (discrete x28523))
;;; y28573 -&gt; (fn [x28558 x28551 x28563 x28564 x28526 x28557 x28552] (nth [(normal x28552 (/ (sqrt x28551))) (normal x28558 (/ (sqrt x28557))) (normal x28564 (/ (sqrt x28563)))] x28526))
;;; x28523 -&gt; (fn [] (dirichlet [1.0 1.0 1.0]))
;;; x28541 -&gt; (fn [x28523] (discrete x28523))
;;; x28538 -&gt; (fn [x28523] (discrete x28523))
;;; x28551 -&gt; (fn [] (gamma 1.0 1.0))
;;; x28558 -&gt; (fn [x28557] (normal 0.0 x28557))
;;; x28552 -&gt; (fn [x28551] (normal 0.0 x28551))
;;; y28608 -&gt; (fn [x28558 x28551 x28541 x28563 x28564 x28557 x28552] (nth [(normal x28552 (/ (sqrt x28551))) (normal x28558 (/ (sqrt x28557))) (normal x28564 (/ (sqrt x28563)))] x28541))
;;; x28563 -&gt; (fn [] (gamma 1.0 1.0))
;;; x28557 -&gt; (fn [] (gamma 1.0 1.0))
;;; x28532 -&gt; (fn [x28523] (discrete x28523))
;;; y28601 -&gt; (fn [x28558 x28551 x28563 x28564 x28557 x28552 x28538] (nth [(normal x28552 (/ (sqrt x28551))) (normal x28558 (/ (sqrt x28557))) (normal x28564 (/ (sqrt x28563)))] x28538))
;;; x28535 -&gt; (fn [x28523] (discrete x28523))
;;; y28587 -&gt; (fn [x28558 x28551 x28563 x28532 x28564 x28557 x28552] (nth [(normal x28552 (/ (sqrt x28551))) (normal x28558 (/ (sqrt x28557))) (normal x28564 (/ (sqrt x28563)))] x28532))
;;; x28529 -&gt; (fn [x28523] (discrete x28523))
;;; 
;;; Observed values O:
;;; y28573 -&gt; 1.1
;;; y28580 -&gt; 2.1
;;; y28587 -&gt; 2.0
;;; y28594 -&gt; 1.9
;;; y28601 -&gt; 0.0
;;; y28608 -&gt; -0.1
;;; y28615 -&gt; -0.05
;;; 
;;; Number of arcs: 59
;;; 
;; <-
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x28526</span>","value":"x28526"},{"type":"html","content":"<span class='clj-symbol'>x28529</span>","value":"x28529"},{"type":"html","content":"<span class='clj-symbol'>x28532</span>","value":"x28532"},{"type":"html","content":"<span class='clj-symbol'>x28535</span>","value":"x28535"},{"type":"html","content":"<span class='clj-symbol'>x28538</span>","value":"x28538"},{"type":"html","content":"<span class='clj-symbol'>x28541</span>","value":"x28541"},{"type":"html","content":"<span class='clj-symbol'>x28544</span>","value":"x28544"}],"value":"[x28526 x28529 x28532 x28535 x28538 x28541 x28544]"}
;; <=

;; **
;;; Q: did this compile correctly?
;;; 
;;; As a sanity check -- how many arcs should there be? If we have 5 data points and three clusters,
;;; 
;;; * each latent z samples from a discrete distribution; 5 arcs total
;;; * each observed y depends on its z, and all three mean/variance pairs; 5 x (3x2 + 1) = 35 arcs total
;;; * each variance depends on its mean, for 3 additional arcs
;;; 
;;; That makes 43 total.
;;; 
;;; If we have 7 data points, then there are 7 + 7x(3x2+1) + 3 = 59.
;;; 
;; **

;; **
;;; ## Multi-layer perceptron
;;; 
;; **

;; @@
(def latent-dim 2)

(def hidden-dim 10)

(def output-dim 5)

(require '[clojure.core.matrix :as mat :refer [mmul add mul div sub]])

(def nn-model
  (foppl-query
    (defn append-gaussian [_ v]
      (conj v (sample (normal 0.0 1.0))))
    
    (defn make-latent-vector [_]
      (loop latent-dim [] append-gaussian))

    (defn make-hidden-vector [_]
      (loop hidden-dim [] append-gaussian))

    (defn make-output-vector [_]
      (loop output-dim [] append-gaussian))
    
    (defn append-latent-vector [_ M]
      (conj M (make-latent-vector)))

    (defn append-hidden-vector [_ M]
      (conj M (make-hidden-vector)))

    (defn append-output-vector [_ M]
      (conj M (make-output-vector)))

    (defn relu [v]
      (mul (mat/ge v 0.0) v))
    
    (defn sigmoid [v]
      (div 1.0 (add 1.0 (mat/exp (sub 0.0 v)))))
    
    (defn append-flip [i v p]
      (conj v (sample (flip (nth p i)))))
    
    (let [z (make-latent-vector)
          
          ;; first: hidden layer
          W (loop hidden-dim [] append-latent-vector)
          b (make-hidden-vector)
          h (relu (add (mmul W z) b))
      
          ;; output layer	
          V (loop output-dim [] append-hidden-vector)
          c (make-output-vector)]
      (loop output-dim [] append-flip (sigmoid (add (mmul V h) c))))))


(:body (second nn-model))
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x24867</span>","value":"x24867"},{"type":"html","content":"<span class='clj-symbol'>x24870</span>","value":"x24870"},{"type":"html","content":"<span class='clj-symbol'>x24873</span>","value":"x24873"},{"type":"html","content":"<span class='clj-symbol'>x24876</span>","value":"x24876"},{"type":"html","content":"<span class='clj-symbol'>x24879</span>","value":"x24879"}],"value":"[x24867 x24870 x24873 x24876 x24879]"}
;; <=

;; @@
(let [trace (foppl/draw-from-prior nn-model)]
  (foppl/get-output nn-model trace))
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-unkown'>false</span>","value":"false"},{"type":"html","content":"<span class='clj-unkown'>false</span>","value":"false"},{"type":"html","content":"<span class='clj-unkown'>false</span>","value":"false"},{"type":"html","content":"<span class='clj-unkown'>false</span>","value":"false"},{"type":"html","content":"<span class='clj-unkown'>false</span>","value":"false"}],"value":"[false false false false false]"}
;; <=

;; @@
(print-graph (first nn-model))

(println "number of arcs:" (count (second (first nn-model))))
;; @@
;; ->
;;; Vertices V: #{x24718 x24765 x24404 x24503 x24837 x24797 x24559 x24785 x24844 x24666 x24450 x24870 x24849 x24864 x24723 x24651 x24688 x24792 x24879 x24693 x24755 x24636 x24656 x24760 x24599 x24419 x24544 x24629 x24579 x24609 x24646 x24438 x24873 x24661 x24534 x24676 x24698 x24807 x24671 x24486 x24641 x24876 x24733 x24614 x24589 x24426 x24574 x24479 x24854 x24827 x24780 x24443 x24431 x24467 x24539 x24594 x24498 x24812 x24708 x24740 x24817 x24549 x24745 x24510 x24713 x24455 x24867 x24522 x24569 x24462 x24681 x24624 x24750 x24491 x24584 x24604 x24414 x24564 x24554 x24802 x24859 x24775 x24728 x24832 x24527 x24474 x24822 x24770 x24619 x24703 x24515 x24409}
;;; 
;;; Arcs A: #{[x24619 x24867] [x24414 x24867] [x24802 x24873] [x24503 x24873] [x24549 x24870] [x24474 x24879] [x24802 x24870] [x24604 x24873] [x24609 x24870] [x24844 x24879] [x24438 x24876] [x24775 x24867] [x24676 x24879] [x24780 x24873] [x24431 x24870] [x24579 x24867] [x24527 x24870] [x24438 x24873] [x24414 x24876] [x24426 x24867] [x24797 x24870] [x24404 x24873] [x24589 x24876] [x24515 x24867] [x24474 x24873] [x24802 x24876] [x24629 x24873] [x24837 x24867] [x24837 x24879] [x24629 x24870] [x24827 x24870] [x24539 x24876] [x24474 x24876] [x24837 x24873] [x24728 x24879] [x24832 x24873] [x24854 x24873] [x24594 x24879] [x24792 x24876] [x24832 x24879] [x24474 x24867] [x24864 x24870] [x24539 x24867] [x24703 x24879] [x24688 x24879] [x24827 x24879] [x24503 x24879] [x24740 x24870] [x24431 x24876] [x24760 x24876] [x24750 x24873] [x24822 x24873] [x24619 x24873] [x24455 x24876] [x24770 x24867] [x24409 x24879] [x24641 x24876] [x24661 x24879] [x24723 x24870] [x24498 x24876] [x24409 x24867] [x24515 x24876] [x24713 x24870] [x24844 x24867] [x24723 x24879] [x24817 x24876] [x24534 x24867] [x24849 x24870] [x24609 x24867] [x24681 x24879] [x24549 x24879] [x24414 x24870] [x24450 x24867] [x24698 x24867] [x24780 x24867] [x24765 x24879] [x24708 x24867] [x24827 x24876] [x24486 x24879] [x24564 x24867] [x24676 x24867] [x24491 x24873] [x24688 x24876] [x24584 x24873] [x24656 x24873] [x24698 x24876] [x24646 x24873] [x24549 x24873] [x24419 x24879] [x24688 x24870] [x24651 x24879] [x24491 x24879] [x24527 x24873] [x24554 x24879] [x24491 x24867] [x24426 x24873] [x24832 x24876] [x24792 x24867] [x24619 x24879] [x24486 x24876] [x24740 x24873] [x24656 x24867] [x24534 x24870] [x24775 x24873] [x24604 x24879] [x24589 x24873] [x24404 x24876] [x24703 x24867] [x24522 x24870] [x24574 x24876] [x24693 x24873] [x24723 x24867] [x24802 x24879] [x24474 x24870] [x24522 x24873] [x24859 x24873] [x24837 x24876] [x24785 x24873] [x24491 x24870] [x24770 x24879] [x24584 x24879] [x24733 x24879] [x24676 x24870] [x24419 x24867] [x24404 x24867] [x24864 x24867] [x24574 x24879] [x24693 x24870] [x24579 x24879] [x24599 x24867] [x24797 x24879] [x24609 x24876] [x24693 x24867] [x24651 x24867] [x24792 x24873] [x24438 x24867] [x24713 x24873] [x24666 x24873] [x24450 x24873] [x24765 x24870] [x24849 x24879] [x24584 x24870] [x24443 x24873] [x24467 x24867] [x24713 x24867] [x24629 x24879] [x24619 x24876] [x24646 x24876] [x24614 x24876] [x24661 x24867] [x24671 x24879] [x24443 x24870] [x24760 x24873] [x24419 x24876] [x24624 x24876] [x24854 x24867] [x24609 x24879] [x24455 x24873] [x24812 x24876] [x24579 x24873] [x24569 x24876] [x24765 x24876] [x24646 x24867] [x24564 x24879] [x24629 x24867] [x24450 x24876] [x24486 x24873] [x24802 x24867] [x24817 x24870] [x24750 x24867] [x24817 x24873] [x24797 x24873] [x24438 x24879] [x24671 x24876] [x24832 x24867] [x24775 x24876] [x24414 x24879] [x24775 x24870] [x24849 x24867] [x24822 x24879] [x24467 x24876] [x24755 x24867] [x24426 x24879] [x24419 x24873] [x24629 x24876] [x24479 x24879] [x24462 x24879] [x24681 x24876] [x24785 x24870] [x24510 x24876] [x24544 x24873] [x24785 x24879] [x24450 x24870] [x24609 x24873] [x24515 x24870] [x24688 x24873] [x24584 x24867] [x24656 x24879] [x24760 x24867] [x24539 x24873] [x24559 x24867] [x24703 x24873] [x24503 x24867] [x24713 x24879] [x24431 x24867] [x24455 x24879] [x24745 x24870] [x24419 x24870] [x24646 x24879] [x24589 x24870] [x24569 x24873] [x24656 x24870] [x24594 x24876] [x24755 x24876] [x24467 x24879] [x24698 x24873] [x24486 x24870] [x24564 x24876] [x24544 x24870] [x24641 x24867] [x24708 x24879] [x24604 x24870] [x24527 x24867] [x24614 x24879] [x24636 x24867] [x24559 x24879] [x24733 x24867] [x24750 x24879] [x24409 x24870] [x24569 x24870] [x24443 x24879] [x24817 x24867] [x24770 x24876] [x24688 x24867] [x24832 x24870] [x24713 x24876] [x24544 x24876] [x24641 x24873] [x24661 x24870] [x24498 x24870] [x24864 x24879] [x24604 x24876] [x24574 x24867] [x24574 x24873] [x24599 x24870] [x24750 x24876] [x24636 x24879] [x24661 x24876] [x24614 x24867] [x24554 x24876] [x24522 x24879] [x24569 x24879] [x24849 x24873] [x24510 x24873] [x24467 x24870] [x24854 x24879] [x24574 x24870] [x24656 x24876] [x24554 x24870] [x24599 x24879] [x24681 x24873] [x24693 x24876] [x24708 x24870] [x24589 x24879] [x24844 x24870] [x24733 x24876] [x24755 x24873] [x24681 x24870] [x24498 x24879] [x24443 x24867] [x24755 x24879] [x24740 x24879] [x24479 x24867] [x24827 x24873] [x24443 x24876] [x24534 x24879] [x24549 x24876] [x24671 x24870] [x24745 x24879] [x24559 x24870] [x24614 x24870] [x24770 x24873] [x24718 x24873] [x24479 x24870] [x24864 x24873] [x24462 x24867] [x24797 x24876] [x24854 x24870] [x24527 x24879] [x24671 x24873] [x24564 x24870] [x24864 x24876] [x24760 x24870] [x24455 x24867] [x24728 x24867] [x24698 x24870] [x24728 x24876] [x24807 x24876] [x24681 x24867] [x24594 x24870] [x24624 x24879] [x24599 x24873] [x24785 x24876] [x24404 x24870] [x24708 x24873] [x24785 x24867] [x24544 x24867] [x24462 x24870] [x24450 x24879] [x24641 x24879] [x24641 x24870] [x24837 x24870] [x24651 x24870] [x24624 x24867] [x24780 x24876] [x24584 x24876] [x24544 x24879] [x24780 x24870] [x24604 x24867] [x24859 x24879] [x24812 x24879] [x24822 x24876] [x24599 x24876] [x24822 x24867] [x24515 x24873] [x24807 x24867] [x24822 x24870] [x24479 x24873] [x24589 x24867] [x24534 x24873] [x24807 x24873] [x24765 x24867] [x24569 x24867] [x24414 x24873] [x24409 x24873] [x24765 x24873] [x24733 x24870] [x24426 x24876] [x24431 x24879] [x24515 x24879] [x24844 x24876] [x24666 x24879] [x24559 x24873] [x24624 x24873] [x24554 x24873] [x24849 x24876] [x24676 x24873] [x24718 x24867] [x24510 x24879] [x24792 x24879] [x24534 x24876] [x24703 x24876] [x24510 x24870] [x24812 x24873] [x24426 x24870] [x24745 x24867] [x24812 x24867] [x24750 x24870] [x24666 x24876] [x24698 x24879] [x24859 x24867] [x24718 x24876] [x24740 x24876] [x24827 x24867] [x24539 x24870] [x24554 x24867] [x24559 x24876] [x24404 x24879] [x24579 x24870] [x24539 x24879] [x24797 x24867] [x24636 x24876] [x24859 x24876] [x24807 x24879] [x24455 x24870] [x24522 x24876] [x24646 x24870] [x24854 x24876] [x24486 x24867] [x24760 x24879] [x24479 x24876] [x24636 x24873] [x24728 x24873] [x24651 x24873] [x24549 x24867] [x24723 x24873] [x24812 x24870] [x24733 x24873] [x24522 x24867] [x24462 x24873] [x24651 x24876] [x24718 x24870] [x24718 x24879] [x24740 x24867] [x24792 x24870] [x24619 x24870] [x24745 x24873] [x24661 x24873] [x24594 x24873] [x24745 x24876] [x24775 x24879] [x24409 x24876] [x24564 x24873] [x24770 x24870] [x24671 x24867] [x24438 x24870] [x24491 x24876] [x24693 x24879] [x24594 x24867] [x24859 x24870] [x24498 x24867] [x24462 x24876] [x24807 x24870] [x24666 x24870] [x24498 x24873] [x24614 x24873] [x24703 x24870] [x24676 x24876] [x24755 x24870] [x24780 x24879] [x24503 x24876] [x24431 x24873] [x24728 x24870] [x24579 x24876] [x24503 x24870] [x24510 x24867] [x24636 x24870] [x24527 x24876] [x24666 x24867] [x24708 x24876] [x24844 x24873] [x24467 x24873] [x24624 x24870] [x24817 x24879] [x24723 x24876]}
;;; 
;;; Conditional densities P:
;;; x24718 -&gt; (fn [] (normal 0.0 1.0))
;;; x24765 -&gt; (fn [] (normal 0.0 1.0))
;;; x24404 -&gt; (fn [] (normal 0.0 1.0))
;;; x24503 -&gt; (fn [] (normal 0.0 1.0))
;;; x24837 -&gt; (fn [] (normal 0.0 1.0))
;;; x24797 -&gt; (fn [] (normal 0.0 1.0))
;;; x24559 -&gt; (fn [] (normal 0.0 1.0))
;;; x24785 -&gt; (fn [] (normal 0.0 1.0))
;;; x24844 -&gt; (fn [] (normal 0.0 1.0))
;;; x24666 -&gt; (fn [] (normal 0.0 1.0))
;;; x24450 -&gt; (fn [] (normal 0.0 1.0))
;;; x24870 -&gt; (fn [x24527 x24775 x24807 x24844 x24661 x24832 x24574 x24419 x24728 x24510 x24671 x24688 x24864 x24614 x24474 x24426 x24827 x24693 x24817 x24589 x24745 x24604 x24534 x24609 x24646 x24733 x24708 x24486 x24849 x24755 x24559 x24515 x24438 x24802 x24780 x24431 x24812 x24564 x24676 x24522 x24837 x24443 x24641 x24740 x24723 x24479 x24770 x24760 x24579 x24584 x24414 x24544 x24624 x24713 x24462 x24467 x24629 x24491 x24765 x24859 x24549 x24854 x24651 x24539 x24619 x24599 x24785 x24455 x24450 x24404 x24698 x24750 x24703 x24409 x24666 x24681 x24636 x24792 x24656 x24569 x24554 x24797 x24498 x24503 x24718 x24594 x24822] (flip (nth (div 1.0 (add 1.0 (mat/exp (sub 0.0 (add (mmul [[x24584 x24589 x24594 x24599 x24604 x24609 x24614 x24619 x24624 x24629] [x24636 x24641 x24646 x24651 x24656 x24661 x24666 x24671 x24676 x24681] [x24688 x24693 x24698 x24703 x24708 x24713 x24718 x24723 x24728 x24733] [x24740 x24745 x24750 x24755 x24760 x24765 x24770 x24775 x24780 x24785] [x24792 x24797 x24802 x24807 x24812 x24817 x24822 x24827 x24832 x24837]] (mul (mat/ge (add (mmul [[x24414 x24419] [x24426 x24431] [x24438 x24443] [x24450 x24455] [x24462 x24467] [x24474 x24479] [x24486 x24491] [x24498 x24503] [x24510 x24515] [x24522 x24527]] [x24404 x24409]) [x24534 x24539 x24544 x24549 x24554 x24559 x24564 x24569 x24574 x24579]) 0.0) (add (mmul [[x24414 x24419] [x24426 x24431] [x24438 x24443] [x24450 x24455] [x24462 x24467] [x24474 x24479] [x24486 x24491] [x24498 x24503] [x24510 x24515] [x24522 x24527]] [x24404 x24409]) [x24534 x24539 x24544 x24549 x24554 x24559 x24564 x24569 x24574 x24579]))) [x24844 x24849 x24854 x24859 x24864]))))) 1)))
;;; x24849 -&gt; (fn [] (normal 0.0 1.0))
;;; x24864 -&gt; (fn [] (normal 0.0 1.0))
;;; x24723 -&gt; (fn [] (normal 0.0 1.0))
;;; x24651 -&gt; (fn [] (normal 0.0 1.0))
;;; x24688 -&gt; (fn [] (normal 0.0 1.0))
;;; x24792 -&gt; (fn [] (normal 0.0 1.0))
;;; x24879 -&gt; (fn [x24527 x24775 x24807 x24844 x24661 x24832 x24574 x24419 x24728 x24510 x24671 x24688 x24864 x24614 x24474 x24426 x24827 x24693 x24817 x24589 x24745 x24604 x24534 x24609 x24646 x24733 x24708 x24486 x24849 x24755 x24559 x24515 x24438 x24802 x24780 x24431 x24812 x24564 x24676 x24522 x24837 x24443 x24641 x24740 x24723 x24479 x24770 x24760 x24579 x24584 x24414 x24544 x24624 x24713 x24462 x24467 x24629 x24491 x24765 x24859 x24549 x24854 x24651 x24539 x24619 x24599 x24785 x24455 x24450 x24404 x24698 x24750 x24703 x24409 x24666 x24681 x24636 x24792 x24656 x24569 x24554 x24797 x24498 x24503 x24718 x24594 x24822] (flip (nth (div 1.0 (add 1.0 (mat/exp (sub 0.0 (add (mmul [[x24584 x24589 x24594 x24599 x24604 x24609 x24614 x24619 x24624 x24629] [x24636 x24641 x24646 x24651 x24656 x24661 x24666 x24671 x24676 x24681] [x24688 x24693 x24698 x24703 x24708 x24713 x24718 x24723 x24728 x24733] [x24740 x24745 x24750 x24755 x24760 x24765 x24770 x24775 x24780 x24785] [x24792 x24797 x24802 x24807 x24812 x24817 x24822 x24827 x24832 x24837]] (mul (mat/ge (add (mmul [[x24414 x24419] [x24426 x24431] [x24438 x24443] [x24450 x24455] [x24462 x24467] [x24474 x24479] [x24486 x24491] [x24498 x24503] [x24510 x24515] [x24522 x24527]] [x24404 x24409]) [x24534 x24539 x24544 x24549 x24554 x24559 x24564 x24569 x24574 x24579]) 0.0) (add (mmul [[x24414 x24419] [x24426 x24431] [x24438 x24443] [x24450 x24455] [x24462 x24467] [x24474 x24479] [x24486 x24491] [x24498 x24503] [x24510 x24515] [x24522 x24527]] [x24404 x24409]) [x24534 x24539 x24544 x24549 x24554 x24559 x24564 x24569 x24574 x24579]))) [x24844 x24849 x24854 x24859 x24864]))))) 4)))
;;; x24693 -&gt; (fn [] (normal 0.0 1.0))
;;; x24755 -&gt; (fn [] (normal 0.0 1.0))
;;; x24636 -&gt; (fn [] (normal 0.0 1.0))
;;; x24656 -&gt; (fn [] (normal 0.0 1.0))
;;; x24760 -&gt; (fn [] (normal 0.0 1.0))
;;; x24599 -&gt; (fn [] (normal 0.0 1.0))
;;; x24419 -&gt; (fn [] (normal 0.0 1.0))
;;; x24544 -&gt; (fn [] (normal 0.0 1.0))
;;; x24629 -&gt; (fn [] (normal 0.0 1.0))
;;; x24579 -&gt; (fn [] (normal 0.0 1.0))
;;; x24609 -&gt; (fn [] (normal 0.0 1.0))
;;; x24646 -&gt; (fn [] (normal 0.0 1.0))
;;; x24438 -&gt; (fn [] (normal 0.0 1.0))
;;; x24873 -&gt; (fn [x24527 x24775 x24807 x24844 x24661 x24832 x24574 x24419 x24728 x24510 x24671 x24688 x24864 x24614 x24474 x24426 x24827 x24693 x24817 x24589 x24745 x24604 x24534 x24609 x24646 x24733 x24708 x24486 x24849 x24755 x24559 x24515 x24438 x24802 x24780 x24431 x24812 x24564 x24676 x24522 x24837 x24443 x24641 x24740 x24723 x24479 x24770 x24760 x24579 x24584 x24414 x24544 x24624 x24713 x24462 x24467 x24629 x24491 x24765 x24859 x24549 x24854 x24651 x24539 x24619 x24599 x24785 x24455 x24450 x24404 x24698 x24750 x24703 x24409 x24666 x24681 x24636 x24792 x24656 x24569 x24554 x24797 x24498 x24503 x24718 x24594 x24822] (flip (nth (div 1.0 (add 1.0 (mat/exp (sub 0.0 (add (mmul [[x24584 x24589 x24594 x24599 x24604 x24609 x24614 x24619 x24624 x24629] [x24636 x24641 x24646 x24651 x24656 x24661 x24666 x24671 x24676 x24681] [x24688 x24693 x24698 x24703 x24708 x24713 x24718 x24723 x24728 x24733] [x24740 x24745 x24750 x24755 x24760 x24765 x24770 x24775 x24780 x24785] [x24792 x24797 x24802 x24807 x24812 x24817 x24822 x24827 x24832 x24837]] (mul (mat/ge (add (mmul [[x24414 x24419] [x24426 x24431] [x24438 x24443] [x24450 x24455] [x24462 x24467] [x24474 x24479] [x24486 x24491] [x24498 x24503] [x24510 x24515] [x24522 x24527]] [x24404 x24409]) [x24534 x24539 x24544 x24549 x24554 x24559 x24564 x24569 x24574 x24579]) 0.0) (add (mmul [[x24414 x24419] [x24426 x24431] [x24438 x24443] [x24450 x24455] [x24462 x24467] [x24474 x24479] [x24486 x24491] [x24498 x24503] [x24510 x24515] [x24522 x24527]] [x24404 x24409]) [x24534 x24539 x24544 x24549 x24554 x24559 x24564 x24569 x24574 x24579]))) [x24844 x24849 x24854 x24859 x24864]))))) 2)))
;;; x24661 -&gt; (fn [] (normal 0.0 1.0))
;;; x24534 -&gt; (fn [] (normal 0.0 1.0))
;;; x24676 -&gt; (fn [] (normal 0.0 1.0))
;;; x24698 -&gt; (fn [] (normal 0.0 1.0))
;;; x24807 -&gt; (fn [] (normal 0.0 1.0))
;;; x24671 -&gt; (fn [] (normal 0.0 1.0))
;;; x24486 -&gt; (fn [] (normal 0.0 1.0))
;;; x24641 -&gt; (fn [] (normal 0.0 1.0))
;;; x24876 -&gt; (fn [x24527 x24775 x24807 x24844 x24661 x24832 x24574 x24419 x24728 x24510 x24671 x24688 x24864 x24614 x24474 x24426 x24827 x24693 x24817 x24589 x24745 x24604 x24534 x24609 x24646 x24733 x24708 x24486 x24849 x24755 x24559 x24515 x24438 x24802 x24780 x24431 x24812 x24564 x24676 x24522 x24837 x24443 x24641 x24740 x24723 x24479 x24770 x24760 x24579 x24584 x24414 x24544 x24624 x24713 x24462 x24467 x24629 x24491 x24765 x24859 x24549 x24854 x24651 x24539 x24619 x24599 x24785 x24455 x24450 x24404 x24698 x24750 x24703 x24409 x24666 x24681 x24636 x24792 x24656 x24569 x24554 x24797 x24498 x24503 x24718 x24594 x24822] (flip (nth (div 1.0 (add 1.0 (mat/exp (sub 0.0 (add (mmul [[x24584 x24589 x24594 x24599 x24604 x24609 x24614 x24619 x24624 x24629] [x24636 x24641 x24646 x24651 x24656 x24661 x24666 x24671 x24676 x24681] [x24688 x24693 x24698 x24703 x24708 x24713 x24718 x24723 x24728 x24733] [x24740 x24745 x24750 x24755 x24760 x24765 x24770 x24775 x24780 x24785] [x24792 x24797 x24802 x24807 x24812 x24817 x24822 x24827 x24832 x24837]] (mul (mat/ge (add (mmul [[x24414 x24419] [x24426 x24431] [x24438 x24443] [x24450 x24455] [x24462 x24467] [x24474 x24479] [x24486 x24491] [x24498 x24503] [x24510 x24515] [x24522 x24527]] [x24404 x24409]) [x24534 x24539 x24544 x24549 x24554 x24559 x24564 x24569 x24574 x24579]) 0.0) (add (mmul [[x24414 x24419] [x24426 x24431] [x24438 x24443] [x24450 x24455] [x24462 x24467] [x24474 x24479] [x24486 x24491] [x24498 x24503] [x24510 x24515] [x24522 x24527]] [x24404 x24409]) [x24534 x24539 x24544 x24549 x24554 x24559 x24564 x24569 x24574 x24579]))) [x24844 x24849 x24854 x24859 x24864]))))) 3)))
;;; x24733 -&gt; (fn [] (normal 0.0 1.0))
;;; x24614 -&gt; (fn [] (normal 0.0 1.0))
;;; x24589 -&gt; (fn [] (normal 0.0 1.0))
;;; x24426 -&gt; (fn [] (normal 0.0 1.0))
;;; x24574 -&gt; (fn [] (normal 0.0 1.0))
;;; x24479 -&gt; (fn [] (normal 0.0 1.0))
;;; x24854 -&gt; (fn [] (normal 0.0 1.0))
;;; x24827 -&gt; (fn [] (normal 0.0 1.0))
;;; x24780 -&gt; (fn [] (normal 0.0 1.0))
;;; x24443 -&gt; (fn [] (normal 0.0 1.0))
;;; x24431 -&gt; (fn [] (normal 0.0 1.0))
;;; x24467 -&gt; (fn [] (normal 0.0 1.0))
;;; x24539 -&gt; (fn [] (normal 0.0 1.0))
;;; x24594 -&gt; (fn [] (normal 0.0 1.0))
;;; x24498 -&gt; (fn [] (normal 0.0 1.0))
;;; x24812 -&gt; (fn [] (normal 0.0 1.0))
;;; x24708 -&gt; (fn [] (normal 0.0 1.0))
;;; x24740 -&gt; (fn [] (normal 0.0 1.0))
;;; x24817 -&gt; (fn [] (normal 0.0 1.0))
;;; x24549 -&gt; (fn [] (normal 0.0 1.0))
;;; x24745 -&gt; (fn [] (normal 0.0 1.0))
;;; x24510 -&gt; (fn [] (normal 0.0 1.0))
;;; x24713 -&gt; (fn [] (normal 0.0 1.0))
;;; x24455 -&gt; (fn [] (normal 0.0 1.0))
;;; x24867 -&gt; (fn [x24527 x24775 x24807 x24844 x24661 x24832 x24574 x24419 x24728 x24510 x24671 x24688 x24864 x24614 x24474 x24426 x24827 x24693 x24817 x24589 x24745 x24604 x24534 x24609 x24646 x24733 x24708 x24486 x24849 x24755 x24559 x24515 x24438 x24802 x24780 x24431 x24812 x24564 x24676 x24522 x24837 x24443 x24641 x24740 x24723 x24479 x24770 x24760 x24579 x24584 x24414 x24544 x24624 x24713 x24462 x24467 x24629 x24491 x24765 x24859 x24549 x24854 x24651 x24539 x24619 x24599 x24785 x24455 x24450 x24404 x24698 x24750 x24703 x24409 x24666 x24681 x24636 x24792 x24656 x24569 x24554 x24797 x24498 x24503 x24718 x24594 x24822] (flip (nth (div 1.0 (add 1.0 (mat/exp (sub 0.0 (add (mmul [[x24584 x24589 x24594 x24599 x24604 x24609 x24614 x24619 x24624 x24629] [x24636 x24641 x24646 x24651 x24656 x24661 x24666 x24671 x24676 x24681] [x24688 x24693 x24698 x24703 x24708 x24713 x24718 x24723 x24728 x24733] [x24740 x24745 x24750 x24755 x24760 x24765 x24770 x24775 x24780 x24785] [x24792 x24797 x24802 x24807 x24812 x24817 x24822 x24827 x24832 x24837]] (mul (mat/ge (add (mmul [[x24414 x24419] [x24426 x24431] [x24438 x24443] [x24450 x24455] [x24462 x24467] [x24474 x24479] [x24486 x24491] [x24498 x24503] [x24510 x24515] [x24522 x24527]] [x24404 x24409]) [x24534 x24539 x24544 x24549 x24554 x24559 x24564 x24569 x24574 x24579]) 0.0) (add (mmul [[x24414 x24419] [x24426 x24431] [x24438 x24443] [x24450 x24455] [x24462 x24467] [x24474 x24479] [x24486 x24491] [x24498 x24503] [x24510 x24515] [x24522 x24527]] [x24404 x24409]) [x24534 x24539 x24544 x24549 x24554 x24559 x24564 x24569 x24574 x24579]))) [x24844 x24849 x24854 x24859 x24864]))))) 0)))
;;; x24522 -&gt; (fn [] (normal 0.0 1.0))
;;; x24569 -&gt; (fn [] (normal 0.0 1.0))
;;; x24462 -&gt; (fn [] (normal 0.0 1.0))
;;; x24681 -&gt; (fn [] (normal 0.0 1.0))
;;; x24624 -&gt; (fn [] (normal 0.0 1.0))
;;; x24750 -&gt; (fn [] (normal 0.0 1.0))
;;; x24491 -&gt; (fn [] (normal 0.0 1.0))
;;; x24584 -&gt; (fn [] (normal 0.0 1.0))
;;; x24604 -&gt; (fn [] (normal 0.0 1.0))
;;; x24414 -&gt; (fn [] (normal 0.0 1.0))
;;; x24564 -&gt; (fn [] (normal 0.0 1.0))
;;; x24554 -&gt; (fn [] (normal 0.0 1.0))
;;; x24802 -&gt; (fn [] (normal 0.0 1.0))
;;; x24859 -&gt; (fn [] (normal 0.0 1.0))
;;; x24775 -&gt; (fn [] (normal 0.0 1.0))
;;; x24728 -&gt; (fn [] (normal 0.0 1.0))
;;; x24832 -&gt; (fn [] (normal 0.0 1.0))
;;; x24527 -&gt; (fn [] (normal 0.0 1.0))
;;; x24474 -&gt; (fn [] (normal 0.0 1.0))
;;; x24822 -&gt; (fn [] (normal 0.0 1.0))
;;; x24770 -&gt; (fn [] (normal 0.0 1.0))
;;; x24619 -&gt; (fn [] (normal 0.0 1.0))
;;; x24703 -&gt; (fn [] (normal 0.0 1.0))
;;; x24515 -&gt; (fn [] (normal 0.0 1.0))
;;; x24409 -&gt; (fn [] (normal 0.0 1.0))
;;; 
;;; Observed values O:
;;; 
;;; number of arcs: 435
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; **
;;; ## Example model mentioned in Issue #1
;; **

;; @@
(def hmm-if
  (foppl-query

    (defn data [n]
      (let [points (vector 0.9 0.8 -0.7 -0.5 -0.025
                           5.0 2.0 0.1 0.0 0.13
                           0.45 6.0 0.2 0.3 -1.0 -1.0)]
        (get points n)))

    (defn hmm-step [n states]
      (let [cur-state (last states)]
        (if (< cur-state 0.)
          (let [next-state (sample (normal 1.0 2.0))]
            (observe (normal 0.0 1.0) (data n))
            (conj states next-state))
          (let [next-state (sample (normal -1.0 2.0))]
            (observe (normal 0.0 1.0) (data n))
            (conj states next-state)))))

    ;; Main Loop through the data
    (let [init-state (sample (normal 0. 5.))]
      (loop 2 (vector init-state) hmm-step))))

(print-graph (first hmm-if))
;; @@
;; ->
;;; Vertices V: #{y29889 y29875 x29882 x29894 x29868 y29901 x29856 y29863 x29849}
;;; 
;;; Arcs A: #{[x29856 y29889] [x29849 y29863] [x29856 y29901] [x29849 y29875]}
;;; 
;;; Conditional densities P:
;;; y29889 -&gt; (fn [x29856] (if (&lt; x29856 0.0) (normal 0.0 1.0)))
;;; y29875 -&gt; (fn [x29849] (if (not (&lt; x29849 0.0)) (normal 0.0 1.0)))
;;; x29882 -&gt; (fn [] (normal 1.0 2.0))
;;; x29894 -&gt; (fn [] (normal -1.0 2.0))
;;; x29868 -&gt; (fn [] (normal -1.0 2.0))
;;; y29901 -&gt; (fn [x29856] (if (not (&lt; x29856 0.0)) (normal 0.0 1.0)))
;;; x29856 -&gt; (fn [] (normal 1.0 2.0))
;;; y29863 -&gt; (fn [x29849] (if (&lt; x29849 0.0) (normal 0.0 1.0)))
;;; x29849 -&gt; (fn [] (normal 0.0 5.0))
;;; 
;;; Observed values O:
;;; y29863 -&gt; 0.9
;;; y29875 -&gt; 0.9
;;; y29889 -&gt; 0.8
;;; y29901 -&gt; 0.8
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; **
;;;       http://www.openbugs.net/Examples/Seeds.html
;;;       
;;;       
;;;       data
;;;       
;;;       list(r = c(10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3),
;;;       n = c(39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7),
;;;       x1 = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
;;;       x2 = c(0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1),
;;;       N = 21)
;;;       
;;;       inits
;;;       
;;;       list(alpha0 = 0, alpha1 = 0, alpha2 = 0, alpha12 = 0, tau = 10)
;;;       
;;;       
;;;       model
;;;       {
;;;          for( i in 1 : N ) {
;;;             r[i] ~ dbin(p[i],n[i])
;;;             b[i] ~ dnorm(0.0,tau)
;;;             logit(p[i]) <- alpha0 + alpha1 * x1[i] + alpha2 * x2[i] + 
;;;                alpha12 * x1[i] * x2[i] + b[i]
;;;          }
;;;          alpha0 ~ dnorm(0.0,1.0E-6)
;;;          alpha1 ~ dnorm(0.0,1.0E-6)
;;;          alpha2 ~ dnorm(0.0,1.0E-6)
;;;          alpha12 ~ dnorm(0.0,1.0E-6)
;;;          tau ~ dgamma(0.001,0.001)
;;;          sigma <- 1 / sqrt(tau)
;;;       }
;; **

;; @@
(def bugs-seeds
  (foppl-query
    (defn data [] 
      [[10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3]    ;r
       [39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7] ;n
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]                   ;x1
       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])                 ;x2
    
    (defn r  [i] (nth (nth (data) 0) i))
    (defn n  [i] (nth (nth (data) 1) i))
    (defn x1 [i] (nth (nth (data) 2) i))
    (defn x2 [i] (nth (nth (data) 3) i))
    
    (defn sigmoid [x] (/ 1. (+ 1 (exp (- x)))))
    
    (defn odds-and-outcome [i _ alpha0 alpha1 alpha2 alpha12 b-dist]
      (let [b (sample b-dist)
            odds (sigmoid (+ alpha0 (* alpha1 (x1 i)) (* alpha2 (x2 i)) (* alpha12 (x1 i) (x2 i)) b))]
        (observe (binomial (n i) odds) (r i))))
    
    (let [alpha0  (sample (normal 0 10))
          alpha1  (sample (normal 0 10))
          alpha2  (sample (normal 0 10))
          alpha12 (sample (normal 0 10))
          tau     (sample (gamma 1 1))
          sigma   (/ 1 (sqrt tau))
          b-dist  (normal 0 sigma)]
      (loop 21 _ odds-and-outcome alpha0 alpha1 alpha2 alpha12 b-dist)
      [alpha0 alpha1 alpha2 alpha12 sigma])))
(let [[G E] bugs-seeds]
  (print-graph G)
  (println "\nNumber of arcs:" (count (second G)))
  (:body E))
;; @@
;; ->
;;; Vertices V: #{x35564 y36012 y35856 x35935 x35857 x35987 x36091 x35779 y35700 y35830 x35558 y35934 x35571 x36013 y35882 x35597 y35960 y35778 y35726 y35908 x35701 y35674 y36116 x35961 x35675 x35805 x35623 y36038 x36039 x35570 y35804 y36090 x35909 x35831 x36065 x35753 y35622 y35986 y35752 x35561 x35727 x35883 y35596 x35649 x35567 y35648 y36064}
;;; 
;;; Arcs A: #{[x35570 x35649] [x35561 y36038] [x35558 y35674] [x35564 y35960] [x35561 y35830] [x35558 y35648] [x35909 y35934] [x35570 x35571] [x35987 y36012] [x35561 y35752] [x35561 y35986] [x35564 y35752] [x35567 y35648] [x36039 y36064] [x35558 y36038] [x35558 y36064] [x35558 y35882] [x35857 y35882] [x35567 y35778] [x35571 y35596] [x35567 y35986] [x35570 x35727] [x35564 y36012] [x35570 x36065] [x35558 y35726] [x35564 y36038] [x35561 y36064] [x35779 y35804] [x35564 y35726] [x35564 y36064] [x35570 x35805] [x35567 y35596] [x35570 x35779] [x35564 y35830] [x35597 y35622] [x35567 y36090] [x35567 y35752] [x35561 y35674] [x35558 y35908] [x35564 y35778] [x35567 y35882] [x36065 y36090] [x35831 y35856] [x35561 y36090] [x35564 y35622] [x35883 y35908] [x35570 x36039] [x35935 y35960] [x35561 y35804] [x35567 y36116] [x35558 y35700] [x35561 y35596] [x35561 y35908] [x35570 x35987] [x35558 y35934] [x35558 y35960] [x35570 x35623] [x35567 y35700] [x35570 x35935] [x35561 y35700] [x35567 y35934] [x35561 y35856] [x35561 y35622] [x35570 x35857] [x35561 y35882] [x35567 y36064] [x35564 y35882] [x35564 y35596] [x35570 x35701] [x35567 y35960] [x35558 y35622] [x35564 y35700] [x35561 y35960] [x35564 y35674] [x35564 y36116] [x35558 y36012] [x35753 y35778] [x35567 y35908] [x35567 y36038] [x35561 y36012] [x35561 y35648] [x35561 y35934] [x35561 y35726] [x35558 y35804] [x35564 y35648] [x35570 x35961] [x35570 x36013] [x35564 y35804] [x35567 y35856] [x35564 y35934] [x35570 x35909] [x35805 y35830] [x35564 y35908] [x35567 y36012] [x35623 y35648] [x35727 y35752] [x35567 y35726] [x35961 y35986] [x35649 y35674] [x35570 x35597] [x35567 y35804] [x35558 y35986] [x35564 y36090] [x35570 x36091] [x35561 y36116] [x35558 y35778] [x35558 y35596] [x35570 x35831] [x35570 x35883] [x35567 y35622] [x36091 y36116] [x35675 y35700] [x35558 y36090] [x35570 x35675] [x35564 y35856] [x35564 y35986] [x35561 y35778] [x35567 y35830] [x35567 y35674] [x35701 y35726] [x35558 y35752] [x36013 y36038] [x35558 y35830] [x35558 y35856] [x35558 y36116] [x35570 x35753]}
;;; 
;;; Conditional densities P:
;;; x35564 -&gt; (fn [] (normal 0 10))
;;; y36012 -&gt; (fn [x35567 x35987 x35558 x35564 x35561] (binomial 12 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 1) (* x35564 1) (* x35567 1 1) x35987)))))))
;;; y35856 -&gt; (fn [x35567 x35558 x35564 x35831 x35561] (binomial 13 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 0) (* x35564 1) (* x35567 0 1) x35831)))))))
;;; x35935 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; x35857 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; x35987 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; x36091 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; x35779 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; y35700 -&gt; (fn [x35567 x35558 x35564 x35675 x35561] (binomial 39 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 0) (* x35564 0) (* x35567 0 0) x35675)))))))
;;; y35830 -&gt; (fn [x35567 x35558 x35564 x35805 x35561] (binomial 79 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 0) (* x35564 1) (* x35567 0 1) x35805)))))))
;;; x35558 -&gt; (fn [] (normal 0 10))
;;; y35934 -&gt; (fn [x35909 x35567 x35558 x35564 x35561] (binomial 28 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 1) (* x35564 0) (* x35567 1 0) x35909)))))))
;;; x35571 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; x36013 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; y35882 -&gt; (fn [x35567 x35857 x35558 x35564 x35561] (binomial 16 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 1) (* x35564 0) (* x35567 1 0) x35857)))))))
;;; x35597 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; y35960 -&gt; (fn [x35567 x35558 x35564 x35935 x35561] (binomial 45 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 1) (* x35564 0) (* x35567 1 0) x35935)))))))
;;; y35778 -&gt; (fn [x35567 x35558 x35753 x35564 x35561] (binomial 72 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 0) (* x35564 1) (* x35567 0 1) x35753)))))))
;;; y35726 -&gt; (fn [x35567 x35701 x35558 x35564 x35561] (binomial 6 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 0) (* x35564 1) (* x35567 0 1) x35701)))))))
;;; y35908 -&gt; (fn [x35883 x35567 x35558 x35564 x35561] (binomial 30 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 1) (* x35564 0) (* x35567 1 0) x35883)))))))
;;; x35701 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; y35674 -&gt; (fn [x35567 x35649 x35558 x35564 x35561] (binomial 51 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 0) (* x35564 0) (* x35567 0 0) x35649)))))))
;;; y36116 -&gt; (fn [x35567 x36091 x35558 x35564 x35561] (binomial 7 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 1) (* x35564 1) (* x35567 1 1) x36091)))))))
;;; x35961 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; x35675 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; x35805 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; x35623 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; y36038 -&gt; (fn [x35567 x35558 x36013 x35564 x35561] (binomial 41 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 1) (* x35564 1) (* x35567 1 1) x36013)))))))
;;; x36039 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; x35570 -&gt; (fn [] (gamma 1 1))
;;; y35804 -&gt; (fn [x35779 x35567 x35558 x35564 x35561] (binomial 51 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 0) (* x35564 1) (* x35567 0 1) x35779)))))))
;;; y36090 -&gt; (fn [x36065 x35567 x35558 x35564 x35561] (binomial 51 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 1) (* x35564 1) (* x35567 1 1) x36065)))))))
;;; x35909 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; x35831 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; x36065 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; x35753 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; y35622 -&gt; (fn [x35567 x35597 x35558 x35564 x35561] (binomial 62 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 0) (* x35564 0) (* x35567 0 0) x35597)))))))
;;; y35986 -&gt; (fn [x35961 x35567 x35558 x35564 x35561] (binomial 4 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 1) (* x35564 0) (* x35567 1 0) x35961)))))))
;;; y35752 -&gt; (fn [x35567 x35558 x35727 x35564 x35561] (binomial 74 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 0) (* x35564 1) (* x35567 0 1) x35727)))))))
;;; x35561 -&gt; (fn [] (normal 0 10))
;;; x35727 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; x35883 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; y35596 -&gt; (fn [x35571 x35567 x35558 x35564 x35561] (binomial 39 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 0) (* x35564 0) (* x35567 0 0) x35571)))))))
;;; x35649 -&gt; (fn [x35570] (normal 0 (/ 1 (sqrt x35570))))
;;; x35567 -&gt; (fn [] (normal 0 10))
;;; y35648 -&gt; (fn [x35567 x35623 x35558 x35564 x35561] (binomial 81 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 0) (* x35564 0) (* x35567 0 0) x35623)))))))
;;; y36064 -&gt; (fn [x35567 x36039 x35558 x35564 x35561] (binomial 30 (/ 1.0 (+ 1 (exp (- (+ x35558 (* x35561 1) (* x35564 1) (* x35567 1 1) x36039)))))))
;;; 
;;; Observed values O:
;;; y36012 -&gt; 3
;;; y35856 -&gt; 10
;;; y35700 -&gt; 17
;;; y35830 -&gt; 46
;;; y35934 -&gt; 8
;;; y35882 -&gt; 8
;;; y35960 -&gt; 23
;;; y35778 -&gt; 55
;;; y35726 -&gt; 5
;;; y35908 -&gt; 10
;;; y35674 -&gt; 26
;;; y36116 -&gt; 3
;;; y36038 -&gt; 22
;;; y35804 -&gt; 32
;;; y36090 -&gt; 32
;;; y35622 -&gt; 23
;;; y35986 -&gt; 0
;;; y35752 -&gt; 53
;;; y35596 -&gt; 10
;;; y35648 -&gt; 23
;;; y36064 -&gt; 15
;;; 
;;; Number of arcs: 126
;;; 
;; <-
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x35558</span>","value":"x35558"},{"type":"html","content":"<span class='clj-symbol'>x35561</span>","value":"x35561"},{"type":"html","content":"<span class='clj-symbol'>x35564</span>","value":"x35564"},{"type":"html","content":"<span class='clj-symbol'>x35567</span>","value":"x35567"},{"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>/</span>","value":"/"},{"type":"html","content":"<span class='clj-long'>1</span>","value":"1"},{"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>sqrt</span>","value":"sqrt"},{"type":"html","content":"<span class='clj-symbol'>x35570</span>","value":"x35570"}],"value":"(sqrt x35570)"}],"value":"(/ 1 (sqrt x35570))"}],"value":"[x35558 x35561 x35564 x35567 (/ 1 (sqrt x35570))]"}
;; <=

;; @@
(def samples (drop 50000 (take 200000 (gibbs/gibbs-seq bugs-seeds))))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;foppl-test/samples</span>","value":"#'foppl-test/samples"}
;; <=

;; @@
(mean samples)
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>-0.5955195690700112</span>","value":"-0.5955195690700112"},{"type":"html","content":"<span class='clj-double'>0.02196012043774304</span>","value":"0.02196012043774304"},{"type":"html","content":"<span class='clj-double'>1.4873937015534684</span>","value":"1.4873937015534684"},{"type":"html","content":"<span class='clj-double'>-0.9016387933521334</span>","value":"-0.9016387933521334"},{"type":"html","content":"<span class='clj-double'>0.5613682431187473</span>","value":"0.5613682431187473"}],"value":"[-0.5955195690700112 0.02196012043774304 1.4873937015534684 -0.9016387933521334 0.5613682431187473]"}
;; <=

;; @@
(def latent-dim 2)

(def hidden-dim 10)

(def output-dim 5)

(require '[clojure.core.matrix :as mat :refer [mmul add mul div sub]])

(def decoder
  (foppl-query
    
    (defn data []
      [[0 1 0.45 2.32 7.23 -1.40 0.01]
       [1 0 4.45 -3.2 0.78 -9.40 1.11]
       [0 1 8.10 5.13 3.90 -6.31 7.41]])

    (defn datum [i] (nth (data) i))
    (defn code [i] (vector (first (datum i)) (second (datum i))))
    (defn output [i] (rest (rest (datum i))))
    
    (defn append-gaussian-value [_ v]
      (conj v (sample (normal 0.0 1.0))))
    
    (defn append-gaussian-dist [_ v]
      (conj v (normal 0.0 1.0)))
    
    (defn make-latent-vector [_]
      (loop latent-dim [] append-gaussian-value))

    (defn make-hidden-vector [_]
      (loop hidden-dim [] append-gaussian-value))

    (defn make-output-param-vector [_]
      (loop output-dim [] append-gaussian-value))
    
    (defn append-latent-vector [_ M]
      (conj M (make-latent-vector)))

    (defn append-hidden-vector [_ M]
      (conj M (make-hidden-vector)))

    (defn relu [v]
      (mul (mat/ge v 0.0) v))
    
    (defn sigmoid [v]
      (div 1.0 (add 1.0 (mat/exp (sub 0.0 v)))))
    
    (defn append-flip [i v p]
      (conj v (sample (flip (nth p i)))))
    
    (defn observe-dim [d _ i mu]
      (observe (normal (nth mu d) 1) (nth (output i) d)))
    
    (defn observe-datum [i _ W b V c]
      (let [z (code i)
            h (relu (add (mmul W z) b))
            mu (add (mmul V h) c)]
        (loop output-dim _ observe-dim i mu)))

    
    (let [W (loop hidden-dim [] append-latent-vector)
          b (make-hidden-vector)
          V (loop output-dim [] append-hidden-vector)
          c (make-output-param-vector)]
      (loop 3 _ observe-datum W b V c)
      [W b V c])))


(:body (second decoder))
;(let [[G E] decoder]
;  (print-graph G)
;  (println "\nNumber of arcs:" (count (second G)))
;  (:body E))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;foppl-test/decoder</span>","value":"#'foppl-test/decoder"}
;; <=

;; @@
(def samples (drop 50 (take 200 (gibbs/gibbs-seq decoder))))
;; @@

;; @@
(take 1 samples)

;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-lazy-seq'>(</span>","close":"<span class='clj-lazy-seq'>)</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>-0.02055612500408616</span>","value":"-0.02055612500408616"},{"type":"html","content":"<span class='clj-double'>-1.3241204101400728</span>","value":"-1.3241204101400728"}],"value":"[-0.02055612500408616 -1.3241204101400728]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>1.0395450835810358</span>","value":"1.0395450835810358"},{"type":"html","content":"<span class='clj-double'>-1.4517244666430058</span>","value":"-1.4517244666430058"}],"value":"[1.0395450835810358 -1.4517244666430058]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>-0.8098132197714168</span>","value":"-0.8098132197714168"},{"type":"html","content":"<span class='clj-double'>-0.7962349839649538</span>","value":"-0.7962349839649538"}],"value":"[-0.8098132197714168 -0.7962349839649538]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>-0.6952558323092033</span>","value":"-0.6952558323092033"},{"type":"html","content":"<span class='clj-double'>-0.07731058191648929</span>","value":"-0.07731058191648929"}],"value":"[-0.6952558323092033 -0.07731058191648929]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>-1.0037865643977175</span>","value":"-1.0037865643977175"},{"type":"html","content":"<span class='clj-double'>-0.967402579062794</span>","value":"-0.967402579062794"}],"value":"[-1.0037865643977175 -0.967402579062794]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>0.15446330624805665</span>","value":"0.15446330624805665"},{"type":"html","content":"<span class='clj-double'>0.6491509614979977</span>","value":"0.6491509614979977"}],"value":"[0.15446330624805665 0.6491509614979977]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>-0.6379046339854564</span>","value":"-0.6379046339854564"},{"type":"html","content":"<span class='clj-double'>1.7416358317139764</span>","value":"1.7416358317139764"}],"value":"[-0.6379046339854564 1.7416358317139764]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>0.4647208983523618</span>","value":"0.4647208983523618"},{"type":"html","content":"<span class='clj-double'>1.1794856632224802</span>","value":"1.1794856632224802"}],"value":"[0.4647208983523618 1.1794856632224802]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>0.2754723375236402</span>","value":"0.2754723375236402"},{"type":"html","content":"<span class='clj-double'>-1.2937527816276508</span>","value":"-1.2937527816276508"}],"value":"[0.2754723375236402 -1.2937527816276508]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>-0.28019555664928214</span>","value":"-0.28019555664928214"},{"type":"html","content":"<span class='clj-double'>1.1415848537202693</span>","value":"1.1415848537202693"}],"value":"[-0.28019555664928214 1.1415848537202693]"}],"value":"[[-0.02055612500408616 -1.3241204101400728] [1.0395450835810358 -1.4517244666430058] [-0.8098132197714168 -0.7962349839649538] [-0.6952558323092033 -0.07731058191648929] [-1.0037865643977175 -0.967402579062794] [0.15446330624805665 0.6491509614979977] [-0.6379046339854564 1.7416358317139764] [0.4647208983523618 1.1794856632224802] [0.2754723375236402 -1.2937527816276508] [-0.28019555664928214 1.1415848537202693]]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>1.1114460014134222</span>","value":"1.1114460014134222"},{"type":"html","content":"<span class='clj-double'>-0.5544722783572378</span>","value":"-0.5544722783572378"},{"type":"html","content":"<span class='clj-double'>-0.8450085986969456</span>","value":"-0.8450085986969456"},{"type":"html","content":"<span class='clj-double'>0.06489486450971035</span>","value":"0.06489486450971035"},{"type":"html","content":"<span class='clj-double'>1.6177565832839502</span>","value":"1.6177565832839502"},{"type":"html","content":"<span class='clj-double'>0.11506965227208928</span>","value":"0.11506965227208928"},{"type":"html","content":"<span class='clj-double'>0.24635187375652282</span>","value":"0.24635187375652282"},{"type":"html","content":"<span class='clj-double'>0.09786099828043505</span>","value":"0.09786099828043505"},{"type":"html","content":"<span class='clj-double'>-0.8448083689028125</span>","value":"-0.8448083689028125"},{"type":"html","content":"<span class='clj-double'>-0.3700201035040988</span>","value":"-0.3700201035040988"}],"value":"[1.1114460014134222 -0.5544722783572378 -0.8450085986969456 0.06489486450971035 1.6177565832839502 0.11506965227208928 0.24635187375652282 0.09786099828043505 -0.8448083689028125 -0.3700201035040988]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>0.5250455213333128</span>","value":"0.5250455213333128"},{"type":"html","content":"<span class='clj-double'>-0.021085726089995874</span>","value":"-0.021085726089995874"},{"type":"html","content":"<span class='clj-double'>-0.3972210416470679</span>","value":"-0.3972210416470679"},{"type":"html","content":"<span class='clj-double'>0.28553023854350335</span>","value":"0.28553023854350335"},{"type":"html","content":"<span class='clj-double'>1.3570954653872103</span>","value":"1.3570954653872103"},{"type":"html","content":"<span class='clj-double'>0.29879106870803734</span>","value":"0.29879106870803734"},{"type":"html","content":"<span class='clj-double'>-0.09857954906993144</span>","value":"-0.09857954906993144"},{"type":"html","content":"<span class='clj-double'>1.1945471356958344</span>","value":"1.1945471356958344"},{"type":"html","content":"<span class='clj-double'>1.2319565274933215</span>","value":"1.2319565274933215"},{"type":"html","content":"<span class='clj-double'>-1.2127166686647435</span>","value":"-1.2127166686647435"}],"value":"[0.5250455213333128 -0.021085726089995874 -0.3972210416470679 0.28553023854350335 1.3570954653872103 0.29879106870803734 -0.09857954906993144 1.1945471356958344 1.2319565274933215 -1.2127166686647435]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>-1.2834232621610209</span>","value":"-1.2834232621610209"},{"type":"html","content":"<span class='clj-double'>0.21108570588629813</span>","value":"0.21108570588629813"},{"type":"html","content":"<span class='clj-double'>0.9221209498416635</span>","value":"0.9221209498416635"},{"type":"html","content":"<span class='clj-double'>-0.5544925137156309</span>","value":"-0.5544925137156309"},{"type":"html","content":"<span class='clj-double'>-1.203879961252357</span>","value":"-1.203879961252357"},{"type":"html","content":"<span class='clj-double'>0.53825103195809</span>","value":"0.53825103195809"},{"type":"html","content":"<span class='clj-double'>0.6848464794643272</span>","value":"0.6848464794643272"},{"type":"html","content":"<span class='clj-double'>-0.5877542753172696</span>","value":"-0.5877542753172696"},{"type":"html","content":"<span class='clj-double'>-1.0632658054208077</span>","value":"-1.0632658054208077"},{"type":"html","content":"<span class='clj-double'>0.6204747711762173</span>","value":"0.6204747711762173"}],"value":"[-1.2834232621610209 0.21108570588629813 0.9221209498416635 -0.5544925137156309 -1.203879961252357 0.53825103195809 0.6848464794643272 -0.5877542753172696 -1.0632658054208077 0.6204747711762173]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>-1.587287361649815</span>","value":"-1.587287361649815"},{"type":"html","content":"<span class='clj-double'>-0.3053678961750724</span>","value":"-0.3053678961750724"},{"type":"html","content":"<span class='clj-double'>-1.234379675950229</span>","value":"-1.234379675950229"},{"type":"html","content":"<span class='clj-double'>0.020660583224567257</span>","value":"0.020660583224567257"},{"type":"html","content":"<span class='clj-double'>1.0994935727559734</span>","value":"1.0994935727559734"},{"type":"html","content":"<span class='clj-double'>1.2232872172936673</span>","value":"1.2232872172936673"},{"type":"html","content":"<span class='clj-double'>0.35837571669474616</span>","value":"0.35837571669474616"},{"type":"html","content":"<span class='clj-double'>1.2940959923472588</span>","value":"1.2940959923472588"},{"type":"html","content":"<span class='clj-double'>-0.7474835759421299</span>","value":"-0.7474835759421299"},{"type":"html","content":"<span class='clj-double'>-2.526855695406497</span>","value":"-2.526855695406497"}],"value":"[-1.587287361649815 -0.3053678961750724 -1.234379675950229 0.020660583224567257 1.0994935727559734 1.2232872172936673 0.35837571669474616 1.2940959923472588 -0.7474835759421299 -2.526855695406497]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>0.25555632906622966</span>","value":"0.25555632906622966"},{"type":"html","content":"<span class='clj-double'>-0.45252069945653206</span>","value":"-0.45252069945653206"},{"type":"html","content":"<span class='clj-double'>-0.696279167798132</span>","value":"-0.696279167798132"},{"type":"html","content":"<span class='clj-double'>0.504462548924444</span>","value":"0.504462548924444"},{"type":"html","content":"<span class='clj-double'>-1.1390937867409505</span>","value":"-1.1390937867409505"},{"type":"html","content":"<span class='clj-double'>-1.014379454230783</span>","value":"-1.014379454230783"},{"type":"html","content":"<span class='clj-double'>0.12303476852729288</span>","value":"0.12303476852729288"},{"type":"html","content":"<span class='clj-double'>0.4989588878572605</span>","value":"0.4989588878572605"},{"type":"html","content":"<span class='clj-double'>-0.36485572947090883</span>","value":"-0.36485572947090883"},{"type":"html","content":"<span class='clj-double'>0.4716041511048926</span>","value":"0.4716041511048926"}],"value":"[0.25555632906622966 -0.45252069945653206 -0.696279167798132 0.504462548924444 -1.1390937867409505 -1.014379454230783 0.12303476852729288 0.4989588878572605 -0.36485572947090883 0.4716041511048926]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>0.4285473806512859</span>","value":"0.4285473806512859"},{"type":"html","content":"<span class='clj-double'>0.699303221283224</span>","value":"0.699303221283224"},{"type":"html","content":"<span class='clj-double'>-0.3439246871798779</span>","value":"-0.3439246871798779"},{"type":"html","content":"<span class='clj-double'>0.7311139845519726</span>","value":"0.7311139845519726"},{"type":"html","content":"<span class='clj-double'>0.49774134312923135</span>","value":"0.49774134312923135"},{"type":"html","content":"<span class='clj-double'>0.5616897947752527</span>","value":"0.5616897947752527"},{"type":"html","content":"<span class='clj-double'>-0.042072575835842695</span>","value":"-0.042072575835842695"},{"type":"html","content":"<span class='clj-double'>-1.0364467894366782</span>","value":"-1.0364467894366782"},{"type":"html","content":"<span class='clj-double'>0.6513110968330368</span>","value":"0.6513110968330368"},{"type":"html","content":"<span class='clj-double'>0.22211838479006416</span>","value":"0.22211838479006416"}],"value":"[0.4285473806512859 0.699303221283224 -0.3439246871798779 0.7311139845519726 0.49774134312923135 0.5616897947752527 -0.042072575835842695 -1.0364467894366782 0.6513110968330368 0.22211838479006416]"}],"value":"[[0.5250455213333128 -0.021085726089995874 -0.3972210416470679 0.28553023854350335 1.3570954653872103 0.29879106870803734 -0.09857954906993144 1.1945471356958344 1.2319565274933215 -1.2127166686647435] [-1.2834232621610209 0.21108570588629813 0.9221209498416635 -0.5544925137156309 -1.203879961252357 0.53825103195809 0.6848464794643272 -0.5877542753172696 -1.0632658054208077 0.6204747711762173] [-1.587287361649815 -0.3053678961750724 -1.234379675950229 0.020660583224567257 1.0994935727559734 1.2232872172936673 0.35837571669474616 1.2940959923472588 -0.7474835759421299 -2.526855695406497] [0.25555632906622966 -0.45252069945653206 -0.696279167798132 0.504462548924444 -1.1390937867409505 -1.014379454230783 0.12303476852729288 0.4989588878572605 -0.36485572947090883 0.4716041511048926] [0.4285473806512859 0.699303221283224 -0.3439246871798779 0.7311139845519726 0.49774134312923135 0.5616897947752527 -0.042072575835842695 -1.0364467894366782 0.6513110968330368 0.22211838479006416]]"},{"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-double'>-0.40127618702940976</span>","value":"-0.40127618702940976"},{"type":"html","content":"<span class='clj-double'>0.4498967053399345</span>","value":"0.4498967053399345"},{"type":"html","content":"<span class='clj-double'>0.7681927536797599</span>","value":"0.7681927536797599"},{"type":"html","content":"<span class='clj-double'>0.5615624929119407</span>","value":"0.5615624929119407"},{"type":"html","content":"<span class='clj-double'>-1.0494025465259798</span>","value":"-1.0494025465259798"}],"value":"[-0.40127618702940976 0.4498967053399345 0.7681927536797599 0.5615624929119407 -1.0494025465259798]"}],"value":"[[[-0.02055612500408616 -1.3241204101400728] [1.0395450835810358 -1.4517244666430058] [-0.8098132197714168 -0.7962349839649538] [-0.6952558323092033 -0.07731058191648929] [-1.0037865643977175 -0.967402579062794] [0.15446330624805665 0.6491509614979977] [-0.6379046339854564 1.7416358317139764] [0.4647208983523618 1.1794856632224802] [0.2754723375236402 -1.2937527816276508] [-0.28019555664928214 1.1415848537202693]] [1.1114460014134222 -0.5544722783572378 -0.8450085986969456 0.06489486450971035 1.6177565832839502 0.11506965227208928 0.24635187375652282 0.09786099828043505 -0.8448083689028125 -0.3700201035040988] [[0.5250455213333128 -0.021085726089995874 -0.3972210416470679 0.28553023854350335 1.3570954653872103 0.29879106870803734 -0.09857954906993144 1.1945471356958344 1.2319565274933215 -1.2127166686647435] [-1.2834232621610209 0.21108570588629813 0.9221209498416635 -0.5544925137156309 -1.203879961252357 0.53825103195809 0.6848464794643272 -0.5877542753172696 -1.0632658054208077 0.6204747711762173] [-1.587287361649815 -0.3053678961750724 -1.234379675950229 0.020660583224567257 1.0994935727559734 1.2232872172936673 0.35837571669474616 1.2940959923472588 -0.7474835759421299 -2.526855695406497] [0.25555632906622966 -0.45252069945653206 -0.696279167798132 0.504462548924444 -1.1390937867409505 -1.014379454230783 0.12303476852729288 0.4989588878572605 -0.36485572947090883 0.4716041511048926] [0.4285473806512859 0.699303221283224 -0.3439246871798779 0.7311139845519726 0.49774134312923135 0.5616897947752527 -0.042072575835842695 -1.0364467894366782 0.6513110968330368 0.22211838479006416]] [-0.40127618702940976 0.4498967053399345 0.7681927536797599 0.5615624929119407 -1.0494025465259798]]"}],"value":"([[[-0.02055612500408616 -1.3241204101400728] [1.0395450835810358 -1.4517244666430058] [-0.8098132197714168 -0.7962349839649538] [-0.6952558323092033 -0.07731058191648929] [-1.0037865643977175 -0.967402579062794] [0.15446330624805665 0.6491509614979977] [-0.6379046339854564 1.7416358317139764] [0.4647208983523618 1.1794856632224802] [0.2754723375236402 -1.2937527816276508] [-0.28019555664928214 1.1415848537202693]] [1.1114460014134222 -0.5544722783572378 -0.8450085986969456 0.06489486450971035 1.6177565832839502 0.11506965227208928 0.24635187375652282 0.09786099828043505 -0.8448083689028125 -0.3700201035040988] [[0.5250455213333128 -0.021085726089995874 -0.3972210416470679 0.28553023854350335 1.3570954653872103 0.29879106870803734 -0.09857954906993144 1.1945471356958344 1.2319565274933215 -1.2127166686647435] [-1.2834232621610209 0.21108570588629813 0.9221209498416635 -0.5544925137156309 -1.203879961252357 0.53825103195809 0.6848464794643272 -0.5877542753172696 -1.0632658054208077 0.6204747711762173] [-1.587287361649815 -0.3053678961750724 -1.234379675950229 0.020660583224567257 1.0994935727559734 1.2232872172936673 0.35837571669474616 1.2940959923472588 -0.7474835759421299 -2.526855695406497] [0.25555632906622966 -0.45252069945653206 -0.696279167798132 0.504462548924444 -1.1390937867409505 -1.014379454230783 0.12303476852729288 0.4989588878572605 -0.36485572947090883 0.4716041511048926] [0.4285473806512859 0.699303221283224 -0.3439246871798779 0.7311139845519726 0.49774134312923135 0.5616897947752527 -0.042072575835842695 -1.0364467894366782 0.6513110968330368 0.22211838479006416]] [-0.40127618702940976 0.4498967053399345 0.7681927536797599 0.5615624929119407 -1.0494025465259798]])"}
;; <=

;; @@

;; @@
