;; gorilla-repl.fileformat = 1

;; **
;;; # Testing FOPPL compilation to graph
;; **

;; @@
(ns foppl-test
  (:require [gorilla-plot.core :as plot]
            [foppl.desugar :refer :all]
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
;;; Vertices V: #{x43042 x43037 x43032 x43047}
;;; 
;;; Arcs A: #{[x43042 x43047]}
;;; 
;;; Conditional densities P:
;;; x43032 -&gt; (fn [] (normal 1 1))
;;; x43037 -&gt; (fn [] (normal 0 2))
;;; x43042 -&gt; (fn [] (normal 0 3))
;;; x43047 -&gt; (fn [x43042] (normal x43042 4))
;;; 
;;; Observed values O:
;;; x43047 -&gt; 2; (fn [] true)
;;; 
;; <-
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x43037</span>","value":"x43037"},{"type":"html","content":"<span class='clj-symbol'>x43042</span>","value":"x43042"}],"value":"[x43037 x43042]"}
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
;;; Vertices V: #{x43109 x43104 x43114 x43099}
;;; 
;;; Arcs A: #{[x43109 x43114]}
;;; 
;;; Conditional densities P:
;;; x43099 -&gt; (fn [] (normal 1 1))
;;; x43104 -&gt; (fn [] (normal 0 2))
;;; x43109 -&gt; (fn [] (normal 0 3))
;;; x43114 -&gt; (fn [x43109] (normal x43109 4))
;;; 
;;; Observed values O:
;;; x43114 -&gt; 2; (fn [] true)
;;; 
;; <-
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"list-like","open":"<span class='clj-list'>(</span>","close":"<span class='clj-list'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>/</span>","value":"/"},{"type":"html","content":"<span class='clj-symbol'>x43104</span>","value":"x43104"},{"type":"html","content":"<span class='clj-long'>2</span>","value":"2"}],"value":"(/ x43104 2)"},{"type":"html","content":"<span class='clj-symbol'>x43109</span>","value":"x43109"}],"value":"[(/ x43104 2) x43109]"}
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
;;; Vertices V: #{x43165}
;;; 
;;; Arcs A: #{}
;;; 
;;; Conditional densities P:
;;; x43165 -&gt; (fn [] (normal 0 1))
;;; 
;;; Observed values O:
;;; 
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-symbol'>x43165</span>","value":"x43165"}
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
;;; Vertices V: #{x43187 x43186}
;;; 
;;; Arcs A: #{[x43186 x43187]}
;;; 
;;; Conditional densities P:
;;; x43186 -&gt; (fn [] (normal 0 1))
;;; x43187 -&gt; (fn [x43186] (normal x43186 1))
;;; 
;;; Observed values O:
;;; 
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-symbol'>x43187</span>","value":"x43187"}
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
;;; Vertices V: #{x43215 x43217 x43216}
;;; 
;;; Arcs A: #{[x43215 x43217] [x43215 x43216] [x43216 x43217]}
;;; 
;;; Conditional densities P:
;;; x43215 -&gt; (fn [] (beta 1 1))
;;; x43216 -&gt; (fn [x43215] (beta (exp x43215) 1))
;;; x43217 -&gt; (fn [x43216 x43215] (bernoulli (* x43216 x43215)))
;;; 
;;; Observed values O:
;;; x43217 -&gt; 1; (fn [] true)
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-symbol'>x43215</span>","value":"x43215"}
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
;;; Vertices V: #{x43273 x43394 x43369 x43266 x43344 x43319 x43294 x43263}
;;; 
;;; Arcs A: #{[x43266 x43344] [x43263 x43319] [x43266 x43273] [x43263 x43273] [x43266 x43394] [x43263 x43394] [x43263 x43369] [x43266 x43294] [x43266 x43319] [x43263 x43294] [x43263 x43344] [x43266 x43369]}
;;; 
;;; Conditional densities P:
;;; x43263 -&gt; (fn [] (normal 0.0 10.0))
;;; x43266 -&gt; (fn [] (normal 0.0 10.0))
;;; x43273 -&gt; (fn [x43263 x43266] (normal (+ (* x43263 1.0) x43266) 1.0))
;;; x43294 -&gt; (fn [x43263 x43266] (normal (+ (* x43263 2.0) x43266) 1.0))
;;; x43319 -&gt; (fn [x43263 x43266] (normal (+ (* x43263 3.0) x43266) 1.0))
;;; x43344 -&gt; (fn [x43263 x43266] (normal (+ (* x43263 4.0) x43266) 1.0))
;;; x43369 -&gt; (fn [x43263 x43266] (normal (+ (* x43263 5.0) x43266) 1.0))
;;; x43394 -&gt; (fn [x43263 x43266] (normal (+ (* x43263 6.0) x43266) 1.0))
;;; 
;;; Observed values O:
;;; x43273 -&gt; 2.1; (fn [] true)
;;; x43294 -&gt; 3.9; (fn [] true)
;;; x43319 -&gt; 5.3; (fn [] true)
;;; x43344 -&gt; 7.7; (fn [] true)
;;; x43369 -&gt; 10.2; (fn [] true)
;;; x43394 -&gt; 12.9; (fn [] true)
;;; 
;; <-
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x43263</span>","value":"x43263"},{"type":"html","content":"<span class='clj-symbol'>x43266</span>","value":"x43266"}],"value":"[x43263 x43266]"}
;; <=

;; @@
(foppl/draw-from-prior linear-regression)
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-record'>#foppl.core.Trace{</span>","close":"<span class='clj-record'>}</span>","separator":" ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:rdb</span>","value":":rdb"},{"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x43263</span>","value":"x43263"},{"type":"html","content":"<span class='clj-double'>-4.706277638989704</span>","value":"-4.706277638989704"}],"value":"[x43263 -4.706277638989704]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x43266</span>","value":"x43266"},{"type":"html","content":"<span class='clj-double'>-19.026450245352134</span>","value":"-19.026450245352134"}],"value":"[x43266 -19.026450245352134]"}],"value":"{x43263 -4.706277638989704, x43266 -19.026450245352134}"}],"value":"[:rdb {x43263 -4.706277638989704, x43266 -19.026450245352134}]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:ordering</span>","value":":ordering"},{"type":"list-like","open":"<span class='clj-list'>(</span>","close":"<span class='clj-list'>)</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x43263</span>","value":"x43263"},{"type":"html","content":"<span class='clj-symbol'>x43266</span>","value":"x43266"},{"type":"html","content":"<span class='clj-symbol'>x43344</span>","value":"x43344"},{"type":"html","content":"<span class='clj-symbol'>x43319</span>","value":"x43319"},{"type":"html","content":"<span class='clj-symbol'>x43294</span>","value":"x43294"},{"type":"html","content":"<span class='clj-symbol'>x43369</span>","value":"x43369"},{"type":"html","content":"<span class='clj-symbol'>x43394</span>","value":"x43394"},{"type":"html","content":"<span class='clj-symbol'>x43273</span>","value":"x43273"}],"value":"(x43263 x43266 x43344 x43319 x43294 x43369 x43394 x43273)"}],"value":"[:ordering (x43263 x43266 x43344 x43319 x43294 x43369 x43394 x43273)]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-keyword'>:logprob</span>","value":":logprob"},{"type":"html","content":"<span class='clj-double'>-5848.495441515548</span>","value":"-5848.495441515548"}],"value":"[:logprob -5848.495441515548]"}],"value":"#foppl.core.Trace{:rdb {x43263 -4.706277638989704, x43266 -19.026450245352134}, :ordering (x43263 x43266 x43344 x43319 x43294 x43369 x43394 x43273), :logprob -5848.495441515548}"}
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
;;; Vertices V: #{x43984 x43834 x43913 x43864 x43954 x43714 x43613 x43823 x44014 x43943 x43673 x43538 x43624 x43804 x43744 x43733 x43684 x43553 x43793 x43924 x43643 x43564 x43703 x43763 x43594 x44003 x43883 x43654 x43853 x43774 x43894 x43973 x43583}
;;; 
;;; Arcs A: #{[x43763 x43793] [x43913 x43924] [x43943 x43973] [x43883 x43913] [x43553 x43564] [x43733 x43763] [x43673 x43684] [x43793 x43804] [x43913 x43943] [x43583 x43594] [x43823 x43853] [x43643 x43673] [x43853 x43883] [x43973 x44003] [x43673 x43703] [x43643 x43654] [x43793 x43823] [x43973 x43984] [x43943 x43954] [x43583 x43613] [x43853 x43864] [x43703 x43714] [x43763 x43774] [x43703 x43733] [x43733 x43744] [x43823 x43834] [x43538 x43553] [x43553 x43583] [x43613 x43643] [x43613 x43624] [x44003 x44014] [x43883 x43894]}
;;; 
;;; Conditional densities P:
;;; x43984 -&gt; (fn [x43973] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43973))
;;; x43834 -&gt; (fn [x43823] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43823))
;;; x43913 -&gt; (fn [x43883] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43883)))
;;; x43864 -&gt; (fn [x43853] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43853))
;;; x43954 -&gt; (fn [x43943] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43943))
;;; x43714 -&gt; (fn [x43703] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43703))
;;; x43613 -&gt; (fn [x43583] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43583)))
;;; x43823 -&gt; (fn [x43793] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43793)))
;;; x44014 -&gt; (fn [x44003] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x44003))
;;; x43943 -&gt; (fn [x43913] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43913)))
;;; x43673 -&gt; (fn [x43643] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43643)))
;;; x43538 -&gt; (fn [] (discrete [0.3333333333333333 0.3333333333333333 0.3333333333333333]))
;;; x43624 -&gt; (fn [x43613] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43613))
;;; x43804 -&gt; (fn [x43793] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43793))
;;; x43744 -&gt; (fn [x43733] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43733))
;;; x43733 -&gt; (fn [x43703] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43703)))
;;; x43684 -&gt; (fn [x43673] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43673))
;;; x43553 -&gt; (fn [x43538] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43538)))
;;; x43793 -&gt; (fn [x43763] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43763)))
;;; x43924 -&gt; (fn [x43913] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43913))
;;; x43643 -&gt; (fn [x43613] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43613)))
;;; x43564 -&gt; (fn [x43553] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43553))
;;; x43703 -&gt; (fn [x43673] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43673)))
;;; x43763 -&gt; (fn [x43733] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43733)))
;;; x43594 -&gt; (fn [x43583] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43583))
;;; x44003 -&gt; (fn [x43973] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43973)))
;;; x43883 -&gt; (fn [x43853] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43853)))
;;; x43654 -&gt; (fn [x43643] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43643))
;;; x43853 -&gt; (fn [x43823] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43823)))
;;; x43774 -&gt; (fn [x43763] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43763))
;;; x43894 -&gt; (fn [x43883] (nth [(normal -1 1) (normal 1 1) (normal 0 1)] x43883))
;;; x43973 -&gt; (fn [x43943] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43943)))
;;; x43583 -&gt; (fn [x43553] (discrete (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.15 0.15 0.7]] x43553)))
;;; 
;;; Observed values O:
;;; x43984 -&gt; -1.0; (fn [] true)
;;; x43834 -&gt; 0.13; (fn [] true)
;;; x43864 -&gt; 0.45; (fn [] true)
;;; x43954 -&gt; 0.3; (fn [] true)
;;; x43714 -&gt; 5.0; (fn [] true)
;;; x44014 -&gt; -1.0; (fn [] true)
;;; x43624 -&gt; 0.7; (fn [] true)
;;; x43804 -&gt; 0.0; (fn [] true)
;;; x43744 -&gt; 2.0; (fn [] true)
;;; x43684 -&gt; -0.025; (fn [] true)
;;; x43924 -&gt; 0.2; (fn [] true)
;;; x43564 -&gt; 0.9; (fn [] true)
;;; x43594 -&gt; 0.8; (fn [] true)
;;; x43654 -&gt; 0.0; (fn [] true)
;;; x43774 -&gt; 0.1; (fn [] true)
;;; x43894 -&gt; 6.0; (fn [] true)
;;; 
;;; Number of arcs: 32
;;; 
;; <-
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x43538</span>","value":"x43538"},{"type":"html","content":"<span class='clj-symbol'>x43553</span>","value":"x43553"},{"type":"html","content":"<span class='clj-symbol'>x43583</span>","value":"x43583"},{"type":"html","content":"<span class='clj-symbol'>x43613</span>","value":"x43613"},{"type":"html","content":"<span class='clj-symbol'>x43643</span>","value":"x43643"},{"type":"html","content":"<span class='clj-symbol'>x43673</span>","value":"x43673"},{"type":"html","content":"<span class='clj-symbol'>x43703</span>","value":"x43703"},{"type":"html","content":"<span class='clj-symbol'>x43733</span>","value":"x43733"},{"type":"html","content":"<span class='clj-symbol'>x43763</span>","value":"x43763"},{"type":"html","content":"<span class='clj-symbol'>x43793</span>","value":"x43793"},{"type":"html","content":"<span class='clj-symbol'>x43823</span>","value":"x43823"},{"type":"html","content":"<span class='clj-symbol'>x43853</span>","value":"x43853"},{"type":"html","content":"<span class='clj-symbol'>x43883</span>","value":"x43883"},{"type":"html","content":"<span class='clj-symbol'>x43913</span>","value":"x43913"},{"type":"html","content":"<span class='clj-symbol'>x43943</span>","value":"x43943"},{"type":"html","content":"<span class='clj-symbol'>x43973</span>","value":"x43973"},{"type":"html","content":"<span class='clj-symbol'>x44003</span>","value":"x44003"}],"value":"[x43538 x43553 x43583 x43613 x43643 x43673 x43703 x43733 x43763 x43793 x43823 x43853 x43883 x43913 x43943 x43973 x44003]"}
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
;;; Vertices V: #{x44334 x44329 x44339}
;;; 
;;; Arcs A: #{[x44334 x44339]}
;;; 
;;; Conditional densities P:
;;; x44329 -&gt; (fn [] (normal 0 1))
;;; x44334 -&gt; (fn [] (normal 0 1))
;;; x44339 -&gt; (fn [x44334] (normal x44334 1))
;;; 
;;; Observed values O:
;;; x44339 -&gt; 1; (fn [] true)
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
;;; Vertices V: #{x44386 x44383 x44378}
;;; 
;;; Arcs A: #{[x44378 x44386] [x44383 x44386]}
;;; 
;;; Conditional densities P:
;;; x44378 -&gt; (fn [] (normal 0 1))
;;; x44383 -&gt; (fn [] (normal 0 1))
;;; x44386 -&gt; (fn [x44383 x44378] (normal (sum [x44378 x44383]) 1))
;;; 
;;; Observed values O:
;;; x44386 -&gt; 1; (fn [] true)
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
;;; Vertices V: #{x44429 x44425 x44428}
;;; 
;;; Arcs A: #{[x44425 x44429]}
;;; 
;;; Conditional densities P:
;;; x44425 -&gt; (fn [] (normal 0 1))
;;; x44428 -&gt; (fn [] (normal 0 1))
;;; x44429 -&gt; (fn [x44425] (normal x44425 1))
;;; 
;;; Observed values O:
;;; x44429 -&gt; 1; (fn [] true)
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
;;; Vertices V: #{x44494 x44583 x44532 x44562 x44526 x44506 x44500 x44503 x44512 x44497 x44509 x44525 x44576 x44569 x44541 x44548 x44520 x44531 x44519 x44491 x44555}
;;; 
;;; Arcs A: #{[x44526 x44541] [x44491 x44500] [x44491 x44512] [x44525 x44576] [x44532 x44562] [x44531 x44555] [x44491 x44494] [x44494 x44541] [x44519 x44576] [x44531 x44569] [x44532 x44576] [x44519 x44541] [x44531 x44532] [x44497 x44548] [x44520 x44562] [x44519 x44569] [x44526 x44548] [x44520 x44569] [x44503 x44562] [x44512 x44583] [x44520 x44576] [x44532 x44548] [x44525 x44541] [x44509 x44576] [x44520 x44583] [x44531 x44576] [x44491 x44503] [x44500 x44555] [x44532 x44569] [x44520 x44548] [x44491 x44506] [x44520 x44555] [x44525 x44548] [x44526 x44576] [x44525 x44583] [x44525 x44526] [x44526 x44569] [x44531 x44583] [x44526 x44583] [x44525 x44569] [x44519 x44562] [x44520 x44541] [x44526 x44562] [x44491 x44509] [x44525 x44562] [x44525 x44555] [x44519 x44583] [x44519 x44555] [x44531 x44548] [x44532 x44583] [x44491 x44497] [x44506 x44569] [x44531 x44541] [x44526 x44555] [x44519 x44520] [x44532 x44555] [x44531 x44562] [x44519 x44548] [x44532 x44541]}
;;; 
;;; Conditional densities P:
;;; x44494 -&gt; (fn [x44491] (discrete x44491))
;;; x44583 -&gt; (fn [x44520 x44512 x44531 x44525 x44519 x44532 x44526] (nth [(normal x44520 (/ (sqrt x44519))) (normal x44526 (/ (sqrt x44525))) (normal x44532 (/ (sqrt x44531)))] x44512))
;;; x44532 -&gt; (fn [x44531] (normal 0.0 x44531))
;;; x44562 -&gt; (fn [x44520 x44503 x44531 x44525 x44519 x44532 x44526] (nth [(normal x44520 (/ (sqrt x44519))) (normal x44526 (/ (sqrt x44525))) (normal x44532 (/ (sqrt x44531)))] x44503))
;;; x44526 -&gt; (fn [x44525] (normal 0.0 x44525))
;;; x44506 -&gt; (fn [x44491] (discrete x44491))
;;; x44500 -&gt; (fn [x44491] (discrete x44491))
;;; x44503 -&gt; (fn [x44491] (discrete x44491))
;;; x44512 -&gt; (fn [x44491] (discrete x44491))
;;; x44497 -&gt; (fn [x44491] (discrete x44491))
;;; x44509 -&gt; (fn [x44491] (discrete x44491))
;;; x44525 -&gt; (fn [] (gamma 1.0 1.0))
;;; x44576 -&gt; (fn [x44520 x44531 x44525 x44519 x44532 x44509 x44526] (nth [(normal x44520 (/ (sqrt x44519))) (normal x44526 (/ (sqrt x44525))) (normal x44532 (/ (sqrt x44531)))] x44509))
;;; x44569 -&gt; (fn [x44520 x44531 x44525 x44506 x44519 x44532 x44526] (nth [(normal x44520 (/ (sqrt x44519))) (normal x44526 (/ (sqrt x44525))) (normal x44532 (/ (sqrt x44531)))] x44506))
;;; x44541 -&gt; (fn [x44520 x44531 x44525 x44494 x44519 x44532 x44526] (nth [(normal x44520 (/ (sqrt x44519))) (normal x44526 (/ (sqrt x44525))) (normal x44532 (/ (sqrt x44531)))] x44494))
;;; x44548 -&gt; (fn [x44520 x44531 x44497 x44525 x44519 x44532 x44526] (nth [(normal x44520 (/ (sqrt x44519))) (normal x44526 (/ (sqrt x44525))) (normal x44532 (/ (sqrt x44531)))] x44497))
;;; x44520 -&gt; (fn [x44519] (normal 0.0 x44519))
;;; x44531 -&gt; (fn [] (gamma 1.0 1.0))
;;; x44519 -&gt; (fn [] (gamma 1.0 1.0))
;;; x44491 -&gt; (fn [] (dirichlet [1.0 1.0 1.0]))
;;; x44555 -&gt; (fn [x44500 x44520 x44531 x44525 x44519 x44532 x44526] (nth [(normal x44520 (/ (sqrt x44519))) (normal x44526 (/ (sqrt x44525))) (normal x44532 (/ (sqrt x44531)))] x44500))
;;; 
;;; Observed values O:
;;; x44541 -&gt; 1.1; (fn [] true)
;;; x44548 -&gt; 2.1; (fn [] true)
;;; x44555 -&gt; 2.0; (fn [] true)
;;; x44562 -&gt; 1.9; (fn [] true)
;;; x44569 -&gt; 0.0; (fn [] true)
;;; x44576 -&gt; -0.1; (fn [] true)
;;; x44583 -&gt; -0.05; (fn [] true)
;;; 
;;; Number of arcs: 59
;;; 
;; <-
;; =>
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x44494</span>","value":"x44494"},{"type":"html","content":"<span class='clj-symbol'>x44497</span>","value":"x44497"},{"type":"html","content":"<span class='clj-symbol'>x44500</span>","value":"x44500"},{"type":"html","content":"<span class='clj-symbol'>x44503</span>","value":"x44503"},{"type":"html","content":"<span class='clj-symbol'>x44506</span>","value":"x44506"},{"type":"html","content":"<span class='clj-symbol'>x44509</span>","value":"x44509"},{"type":"html","content":"<span class='clj-symbol'>x44512</span>","value":"x44512"}],"value":"[x44494 x44497 x44500 x44503 x44506 x44509 x44512]"}
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
;;; {"type":"list-like","open":"<span class='clj-vector'>[</span>","close":"<span class='clj-vector'>]</span>","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x47474</span>","value":"x47474"},{"type":"html","content":"<span class='clj-symbol'>x47479</span>","value":"x47479"},{"type":"html","content":"<span class='clj-symbol'>x47484</span>","value":"x47484"},{"type":"html","content":"<span class='clj-symbol'>x47489</span>","value":"x47489"},{"type":"html","content":"<span class='clj-symbol'>x47494</span>","value":"x47494"}],"value":"[x47474 x47479 x47484 x47489 x47494]"}
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
;;; Vertices V: #{x47154 x47432 x47474 x47229 x47091 x47308 x47370 x47204 x47031 x47159 x47096 x47286 x47407 x47454 x47390 x47417 x47489 x47184 x47115 x47132 x47442 x47350 x47494 x47224 x47256 x47313 x47303 x47380 x47338 x47024 x47219 x47469 x47246 x47484 x47036 x47048 x47139 x47174 x47199 x47437 x47281 x47164 x47328 x47355 x47365 x47427 x47209 x47189 x47266 x47194 x47108 x47014 x47251 x47103 x47464 x47276 x47318 x47298 x47067 x47072 x47084 x47127 x47019 x47169 x47079 x47412 x47261 x47060 x47120 x47360 x47402 x47323 x47397 x47459 x47043 x47375 x47333 x47449 x47144 x47271 x47293 x47241 x47345 x47149 x47214 x47234 x47385 x47055 x47009 x47422 x47179 x47479}
;;; 
;;; Arcs A: #{[x47043 x47489] [x47293 x47494] [x47318 x47489] [x47469 x47479] [x47390 x47494] [x47459 x47474] [x47350 x47484] [x47079 x47484] [x47120 x47474] [x47412 x47484] [x47043 x47479] [x47169 x47474] [x47251 x47479] [x47031 x47479] [x47084 x47489] [x47036 x47474] [x47164 x47494] [x47084 x47474] [x47407 x47484] [x47139 x47474] [x47115 x47484] [x47174 x47494] [x47209 x47494] [x47159 x47489] [x47014 x47489] [x47308 x47489] [x47328 x47489] [x47199 x47494] [x47014 x47474] [x47380 x47489] [x47096 x47484] [x47019 x47494] [x47072 x47484] [x47159 x47479] [x47437 x47474] [x47014 x47479] [x47397 x47479] [x47139 x47484] [x47009 x47484] [x47360 x47479] [x47318 x47474] [x47338 x47489] [x47333 x47489] [x47328 x47479] [x47313 x47494] [x47048 x47484] [x47204 x47494] [x47333 x47484] [x47043 x47484] [x47036 x47479] [x47437 x47489] [x47345 x47489] [x47442 x47494] [x47432 x47489] [x47031 x47489] [x47345 x47474] [x47048 x47474] [x47360 x47484] [x47328 x47474] [x47323 x47474] [x47048 x47489] [x47219 x47484] [x47144 x47489] [x47214 x47484] [x47184 x47474] [x47209 x47474] [x47024 x47494] [x47199 x47489] [x47234 x47474] [x47036 x47489] [x47019 x47489] [x47194 x47474] [x47442 x47479] [x47261 x47494] [x47281 x47484] [x47091 x47479] [x47276 x47489] [x47159 x47474] [x47318 x47484] [x47469 x47489] [x47385 x47474] [x47096 x47494] [x47246 x47494] [x47437 x47484] [x47459 x47489] [x47169 x47484] [x47276 x47474] [x47350 x47479] [x47139 x47489] [x47432 x47479] [x47108 x47489] [x47120 x47479] [x47385 x47489] [x47108 x47494] [x47251 x47484] [x47345 x47494] [x47355 x47489] [x47469 x47494] [x47194 x47494] [x47164 x47474] [x47402 x47479] [x47072 x47479] [x47308 x47479] [x47204 x47479] [x47442 x47489] [x47271 x47489] [x47169 x47494] [x47390 x47489] [x47229 x47479] [x47286 x47494] [x47241 x47484] [x47293 x47489] [x47464 x47494] [x47355 x47494] [x47132 x47484] [x47209 x47479] [x47120 x47489] [x47308 x47474] [x47417 x47489] [x47009 x47494] [x47159 x47494] [x47338 x47474] [x47370 x47474] [x47132 x47474] [x47229 x47474] [x47179 x47494] [x47164 x47489] [x47380 x47479] [x47043 x47474] [x47079 x47489] [x47024 x47484] [x47385 x47494] [x47407 x47479] [x47072 x47494] [x47072 x47474] [x47204 x47474] [x47024 x47479] [x47360 x47489] [x47276 x47484] [x47234 x47489] [x47464 x47489] [x47174 x47484] [x47422 x47489] [x47365 x47484] [x47385 x47479] [x47375 x47479] [x47084 x47484] [x47209 x47484] [x47055 x47494] [x47229 x47489] [x47251 x47474] [x47127 x47474] [x47459 x47494] [x47427 x47489] [x47323 x47479] [x47214 x47489] [x47313 x47484] [x47127 x47489] [x47096 x47479] [x47422 x47484] [x47293 x47484] [x47154 x47489] [x47303 x47494] [x47333 x47474] [x47365 x47474] [x47219 x47494] [x47370 x47489] [x47469 x47484] [x47459 x47479] [x47390 x47479] [x47224 x47494] [x47036 x47484] [x47229 x47494] [x47115 x47494] [x47031 x47474] [x47067 x47484] [x47454 x47494] [x47454 x47484] [x47286 x47474] [x47084 x47479] [x47422 x47474] [x47174 x47474] [x47184 x47494] [x47303 x47489] [x47067 x47489] [x47397 x47489] [x47091 x47484] [x47256 x47474] [x47060 x47484] [x47271 x47479] [x47019 x47474] [x47355 x47474] [x47271 x47494] [x47079 x47479] [x47009 x47489] [x47096 x47489] [x47449 x47479] [x47103 x47484] [x47407 x47474] [x47132 x47479] [x47338 x47479] [x47251 x47494] [x47031 x47484] [x47303 x47484] [x47417 x47474] [x47024 x47489] [x47454 x47479] [x47224 x47474] [x47179 x47489] [x47261 x47479] [x47281 x47494] [x47115 x47479] [x47422 x47479] [x47256 x47494] [x47019 x47484] [x47214 x47494] [x47194 x47479] [x47199 x47474] [x47412 x47474] [x47149 x47489] [x47298 x47494] [x47199 x47479] [x47169 x47489] [x47043 x47494] [x47437 x47494] [x47103 x47494] [x47060 x47474] [x47241 x47489] [x47084 x47494] [x47271 x47484] [x47219 x47489] [x47154 x47494] [x47427 x47474] [x47229 x47484] [x47333 x47479] [x47375 x47474] [x47407 x47489] [x47442 x47474] [x47246 x47479] [x47390 x47484] [x47189 x47489] [x47091 x47489] [x47375 x47494] [x47031 x47494] [x47432 x47474] [x47365 x47489] [x47365 x47479] [x47009 x47479] [x47154 x47484] [x47375 x47489] [x47204 x47484] [x47437 x47479] [x47214 x47474] [x47318 x47494] [x47184 x47484] [x47144 x47494] [x47014 x47484] [x47224 x47489] [x47385 x47484] [x47266 x47489] [x47449 x47489] [x47199 x47484] [x47333 x47494] [x47281 x47479] [x47464 x47484] [x47149 x47484] [x47055 x47484] [x47380 x47484] [x47234 x47484] [x47276 x47494] [x47120 x47494] [x47189 x47474] [x47184 x47479] [x47427 x47479] [x47298 x47484] [x47103 x47474] [x47060 x47479] [x47271 x47474] [x47412 x47479] [x47370 x47484] [x47266 x47474] [x47286 x47489] [x47375 x47484] [x47298 x47474] [x47204 x47489] [x47293 x47479] [x47293 x47474] [x47459 x47484] [x47355 x47479] [x47417 x47494] [x47067 x47479] [x47115 x47474] [x47055 x47489] [x47313 x47479] [x47149 x47494] [x47370 x47479] [x47432 x47494] [x47256 x47484] [x47402 x47484] [x47055 x47479] [x47454 x47489] [x47308 x47484] [x47149 x47474] [x47328 x47494] [x47179 x47474] [x47079 x47494] [x47323 x47494] [x47164 x47484] [x47144 x47479] [x47276 x47479] [x47009 x47474] [x47432 x47484] [x47449 x47494] [x47318 x47479] [x47144 x47484] [x47469 x47474] [x47261 x47484] [x47132 x47489] [x47412 x47494] [x47184 x47489] [x47179 x47479] [x47219 x47474] [x47219 x47479] [x47422 x47494] [x47189 x47494] [x47115 x47489] [x47241 x47479] [x47246 x47484] [x47397 x47494] [x47060 x47494] [x47380 x47494] [x47266 x47494] [x47298 x47489] [x47338 x47494] [x47024 x47474] [x47189 x47484] [x47266 x47479] [x47345 x47479] [x47164 x47479] [x47281 x47489] [x47139 x47494] [x47234 x47479] [x47014 x47494] [x47108 x47479] [x47139 x47479] [x47449 x47474] [x47251 x47489] [x47174 x47479] [x47402 x47489] [x47442 x47484] [x47256 x47489] [x47397 x47474] [x47060 x47489] [x47397 x47484] [x47464 x47474] [x47149 x47479] [x47048 x47494] [x47417 x47479] [x47303 x47479] [x47209 x47489] [x47313 x47474] [x47154 x47479] [x47412 x47489] [x47360 x47494] [x47194 x47484] [x47323 x47484] [x47241 x47474] [x47179 x47484] [x47174 x47489] [x47055 x47474] [x47286 x47479] [x47036 x47494] [x47246 x47489] [x47308 x47494] [x47417 x47484] [x47224 x47479] [x47189 x47479] [x47402 x47474] [x47214 x47479] [x47345 x47484] [x47079 x47474] [x47355 x47484] [x47390 x47474] [x47370 x47494] [x47246 x47474] [x47091 x47494] [x47154 x47474] [x47072 x47489] [x47261 x47489] [x47454 x47474] [x47350 x47474] [x47067 x47494] [x47127 x47494] [x47144 x47474] [x47194 x47489] [x47464 x47479] [x47132 x47494] [x47127 x47484] [x47365 x47494] [x47286 x47484] [x47281 x47474] [x47313 x47489] [x47127 x47479] [x47120 x47484] [x47323 x47489] [x47096 x47474] [x47019 x47479] [x47256 x47479] [x47103 x47479] [x47298 x47479] [x47261 x47474] [x47241 x47494] [x47360 x47474] [x47303 x47474] [x47234 x47494] [x47328 x47484] [x47350 x47494] [x47048 x47479] [x47108 x47474] [x47427 x47494] [x47067 x47474] [x47169 x47479] [x47350 x47489] [x47380 x47474] [x47407 x47494] [x47427 x47484] [x47338 x47484] [x47449 x47484] [x47103 x47489] [x47224 x47484] [x47159 x47484] [x47091 x47474] [x47266 x47484] [x47108 x47484] [x47402 x47494]}
;;; 
;;; Conditional densities P:
;;; x47154 -&gt; (fn [] (normal 0.0 1.0))
;;; x47432 -&gt; (fn [] (normal 0.0 1.0))
;;; x47474 -&gt; (fn [x47298 x47219 x47019 x47390 x47159 x47108 x47417 x47048 x47209 x47055 x47224 x47266 x47072 x47149 x47313 x47338 x47084 x47286 x47204 x47464 x47407 x47031 x47303 x47234 x47144 x47422 x47091 x47345 x47370 x47427 x47380 x47024 x47132 x47014 x47169 x47120 x47293 x47036 x47229 x47009 x47127 x47079 x47459 x47375 x47214 x47323 x47350 x47256 x47328 x47189 x47096 x47276 x47174 x47308 x47154 x47043 x47184 x47103 x47365 x47449 x47246 x47164 x47199 x47261 x47385 x47355 x47397 x47469 x47454 x47318 x47412 x47271 x47115 x47360 x47442 x47139 x47179 x47067 x47060 x47194 x47437 x47333 x47251 x47241 x47432 x47402 x47281] (flip (nth (div 1.0 (add 1.0 (mat/exp (sub 0.0 (add (mmul [[x47189 x47194 x47199 x47204 x47209 x47214 x47219 x47224 x47229 x47234] [x47241 x47246 x47251 x47256 x47261 x47266 x47271 x47276 x47281 x47286] [x47293 x47298 x47303 x47308 x47313 x47318 x47323 x47328 x47333 x47338] [x47345 x47350 x47355 x47360 x47365 x47370 x47375 x47380 x47385 x47390] [x47397 x47402 x47407 x47412 x47417 x47422 x47427 x47432 x47437 x47442]] (mul (mat/ge (add (mmul [[x47019 x47024] [x47031 x47036] [x47043 x47048] [x47055 x47060] [x47067 x47072] [x47079 x47084] [x47091 x47096] [x47103 x47108] [x47115 x47120] [x47127 x47132]] [x47009 x47014]) [x47139 x47144 x47149 x47154 x47159 x47164 x47169 x47174 x47179 x47184]) 0.0) (add (mmul [[x47019 x47024] [x47031 x47036] [x47043 x47048] [x47055 x47060] [x47067 x47072] [x47079 x47084] [x47091 x47096] [x47103 x47108] [x47115 x47120] [x47127 x47132]] [x47009 x47014]) [x47139 x47144 x47149 x47154 x47159 x47164 x47169 x47174 x47179 x47184]))) [x47449 x47454 x47459 x47464 x47469]))))) 0)))
;;; x47229 -&gt; (fn [] (normal 0.0 1.0))
;;; x47091 -&gt; (fn [] (normal 0.0 1.0))
;;; x47308 -&gt; (fn [] (normal 0.0 1.0))
;;; x47370 -&gt; (fn [] (normal 0.0 1.0))
;;; x47204 -&gt; (fn [] (normal 0.0 1.0))
;;; x47031 -&gt; (fn [] (normal 0.0 1.0))
;;; x47159 -&gt; (fn [] (normal 0.0 1.0))
;;; x47096 -&gt; (fn [] (normal 0.0 1.0))
;;; x47286 -&gt; (fn [] (normal 0.0 1.0))
;;; x47407 -&gt; (fn [] (normal 0.0 1.0))
;;; x47454 -&gt; (fn [] (normal 0.0 1.0))
;;; x47390 -&gt; (fn [] (normal 0.0 1.0))
;;; x47417 -&gt; (fn [] (normal 0.0 1.0))
;;; x47489 -&gt; (fn [x47298 x47219 x47019 x47390 x47159 x47108 x47417 x47048 x47209 x47055 x47224 x47266 x47072 x47149 x47313 x47338 x47084 x47286 x47204 x47464 x47407 x47031 x47303 x47234 x47144 x47422 x47091 x47345 x47370 x47427 x47380 x47024 x47132 x47014 x47169 x47120 x47293 x47036 x47229 x47009 x47127 x47079 x47459 x47375 x47214 x47323 x47350 x47256 x47328 x47189 x47096 x47276 x47174 x47308 x47154 x47043 x47184 x47103 x47365 x47449 x47246 x47164 x47199 x47261 x47385 x47355 x47397 x47469 x47454 x47318 x47412 x47271 x47115 x47360 x47442 x47139 x47179 x47067 x47060 x47194 x47437 x47333 x47251 x47241 x47432 x47402 x47281] (flip (nth (div 1.0 (add 1.0 (mat/exp (sub 0.0 (add (mmul [[x47189 x47194 x47199 x47204 x47209 x47214 x47219 x47224 x47229 x47234] [x47241 x47246 x47251 x47256 x47261 x47266 x47271 x47276 x47281 x47286] [x47293 x47298 x47303 x47308 x47313 x47318 x47323 x47328 x47333 x47338] [x47345 x47350 x47355 x47360 x47365 x47370 x47375 x47380 x47385 x47390] [x47397 x47402 x47407 x47412 x47417 x47422 x47427 x47432 x47437 x47442]] (mul (mat/ge (add (mmul [[x47019 x47024] [x47031 x47036] [x47043 x47048] [x47055 x47060] [x47067 x47072] [x47079 x47084] [x47091 x47096] [x47103 x47108] [x47115 x47120] [x47127 x47132]] [x47009 x47014]) [x47139 x47144 x47149 x47154 x47159 x47164 x47169 x47174 x47179 x47184]) 0.0) (add (mmul [[x47019 x47024] [x47031 x47036] [x47043 x47048] [x47055 x47060] [x47067 x47072] [x47079 x47084] [x47091 x47096] [x47103 x47108] [x47115 x47120] [x47127 x47132]] [x47009 x47014]) [x47139 x47144 x47149 x47154 x47159 x47164 x47169 x47174 x47179 x47184]))) [x47449 x47454 x47459 x47464 x47469]))))) 3)))
;;; x47184 -&gt; (fn [] (normal 0.0 1.0))
;;; x47115 -&gt; (fn [] (normal 0.0 1.0))
;;; x47132 -&gt; (fn [] (normal 0.0 1.0))
;;; x47442 -&gt; (fn [] (normal 0.0 1.0))
;;; x47350 -&gt; (fn [] (normal 0.0 1.0))
;;; x47494 -&gt; (fn [x47298 x47219 x47019 x47390 x47159 x47108 x47417 x47048 x47209 x47055 x47224 x47266 x47072 x47149 x47313 x47338 x47084 x47286 x47204 x47464 x47407 x47031 x47303 x47234 x47144 x47422 x47091 x47345 x47370 x47427 x47380 x47024 x47132 x47014 x47169 x47120 x47293 x47036 x47229 x47009 x47127 x47079 x47459 x47375 x47214 x47323 x47350 x47256 x47328 x47189 x47096 x47276 x47174 x47308 x47154 x47043 x47184 x47103 x47365 x47449 x47246 x47164 x47199 x47261 x47385 x47355 x47397 x47469 x47454 x47318 x47412 x47271 x47115 x47360 x47442 x47139 x47179 x47067 x47060 x47194 x47437 x47333 x47251 x47241 x47432 x47402 x47281] (flip (nth (div 1.0 (add 1.0 (mat/exp (sub 0.0 (add (mmul [[x47189 x47194 x47199 x47204 x47209 x47214 x47219 x47224 x47229 x47234] [x47241 x47246 x47251 x47256 x47261 x47266 x47271 x47276 x47281 x47286] [x47293 x47298 x47303 x47308 x47313 x47318 x47323 x47328 x47333 x47338] [x47345 x47350 x47355 x47360 x47365 x47370 x47375 x47380 x47385 x47390] [x47397 x47402 x47407 x47412 x47417 x47422 x47427 x47432 x47437 x47442]] (mul (mat/ge (add (mmul [[x47019 x47024] [x47031 x47036] [x47043 x47048] [x47055 x47060] [x47067 x47072] [x47079 x47084] [x47091 x47096] [x47103 x47108] [x47115 x47120] [x47127 x47132]] [x47009 x47014]) [x47139 x47144 x47149 x47154 x47159 x47164 x47169 x47174 x47179 x47184]) 0.0) (add (mmul [[x47019 x47024] [x47031 x47036] [x47043 x47048] [x47055 x47060] [x47067 x47072] [x47079 x47084] [x47091 x47096] [x47103 x47108] [x47115 x47120] [x47127 x47132]] [x47009 x47014]) [x47139 x47144 x47149 x47154 x47159 x47164 x47169 x47174 x47179 x47184]))) [x47449 x47454 x47459 x47464 x47469]))))) 4)))
;;; x47224 -&gt; (fn [] (normal 0.0 1.0))
;;; x47256 -&gt; (fn [] (normal 0.0 1.0))
;;; x47313 -&gt; (fn [] (normal 0.0 1.0))
;;; x47303 -&gt; (fn [] (normal 0.0 1.0))
;;; x47380 -&gt; (fn [] (normal 0.0 1.0))
;;; x47338 -&gt; (fn [] (normal 0.0 1.0))
;;; x47024 -&gt; (fn [] (normal 0.0 1.0))
;;; x47219 -&gt; (fn [] (normal 0.0 1.0))
;;; x47469 -&gt; (fn [] (normal 0.0 1.0))
;;; x47246 -&gt; (fn [] (normal 0.0 1.0))
;;; x47484 -&gt; (fn [x47298 x47219 x47019 x47390 x47159 x47108 x47417 x47048 x47209 x47055 x47224 x47266 x47072 x47149 x47313 x47338 x47084 x47286 x47204 x47464 x47407 x47031 x47303 x47234 x47144 x47422 x47091 x47345 x47370 x47427 x47380 x47024 x47132 x47014 x47169 x47120 x47293 x47036 x47229 x47009 x47127 x47079 x47459 x47375 x47214 x47323 x47350 x47256 x47328 x47189 x47096 x47276 x47174 x47308 x47154 x47043 x47184 x47103 x47365 x47449 x47246 x47164 x47199 x47261 x47385 x47355 x47397 x47469 x47454 x47318 x47412 x47271 x47115 x47360 x47442 x47139 x47179 x47067 x47060 x47194 x47437 x47333 x47251 x47241 x47432 x47402 x47281] (flip (nth (div 1.0 (add 1.0 (mat/exp (sub 0.0 (add (mmul [[x47189 x47194 x47199 x47204 x47209 x47214 x47219 x47224 x47229 x47234] [x47241 x47246 x47251 x47256 x47261 x47266 x47271 x47276 x47281 x47286] [x47293 x47298 x47303 x47308 x47313 x47318 x47323 x47328 x47333 x47338] [x47345 x47350 x47355 x47360 x47365 x47370 x47375 x47380 x47385 x47390] [x47397 x47402 x47407 x47412 x47417 x47422 x47427 x47432 x47437 x47442]] (mul (mat/ge (add (mmul [[x47019 x47024] [x47031 x47036] [x47043 x47048] [x47055 x47060] [x47067 x47072] [x47079 x47084] [x47091 x47096] [x47103 x47108] [x47115 x47120] [x47127 x47132]] [x47009 x47014]) [x47139 x47144 x47149 x47154 x47159 x47164 x47169 x47174 x47179 x47184]) 0.0) (add (mmul [[x47019 x47024] [x47031 x47036] [x47043 x47048] [x47055 x47060] [x47067 x47072] [x47079 x47084] [x47091 x47096] [x47103 x47108] [x47115 x47120] [x47127 x47132]] [x47009 x47014]) [x47139 x47144 x47149 x47154 x47159 x47164 x47169 x47174 x47179 x47184]))) [x47449 x47454 x47459 x47464 x47469]))))) 2)))
;;; x47036 -&gt; (fn [] (normal 0.0 1.0))
;;; x47048 -&gt; (fn [] (normal 0.0 1.0))
;;; x47139 -&gt; (fn [] (normal 0.0 1.0))
;;; x47174 -&gt; (fn [] (normal 0.0 1.0))
;;; x47199 -&gt; (fn [] (normal 0.0 1.0))
;;; x47437 -&gt; (fn [] (normal 0.0 1.0))
;;; x47281 -&gt; (fn [] (normal 0.0 1.0))
;;; x47164 -&gt; (fn [] (normal 0.0 1.0))
;;; x47328 -&gt; (fn [] (normal 0.0 1.0))
;;; x47355 -&gt; (fn [] (normal 0.0 1.0))
;;; x47365 -&gt; (fn [] (normal 0.0 1.0))
;;; x47427 -&gt; (fn [] (normal 0.0 1.0))
;;; x47209 -&gt; (fn [] (normal 0.0 1.0))
;;; x47189 -&gt; (fn [] (normal 0.0 1.0))
;;; x47266 -&gt; (fn [] (normal 0.0 1.0))
;;; x47194 -&gt; (fn [] (normal 0.0 1.0))
;;; x47108 -&gt; (fn [] (normal 0.0 1.0))
;;; x47014 -&gt; (fn [] (normal 0.0 1.0))
;;; x47251 -&gt; (fn [] (normal 0.0 1.0))
;;; x47103 -&gt; (fn [] (normal 0.0 1.0))
;;; x47464 -&gt; (fn [] (normal 0.0 1.0))
;;; x47276 -&gt; (fn [] (normal 0.0 1.0))
;;; x47318 -&gt; (fn [] (normal 0.0 1.0))
;;; x47298 -&gt; (fn [] (normal 0.0 1.0))
;;; x47067 -&gt; (fn [] (normal 0.0 1.0))
;;; x47072 -&gt; (fn [] (normal 0.0 1.0))
;;; x47084 -&gt; (fn [] (normal 0.0 1.0))
;;; x47127 -&gt; (fn [] (normal 0.0 1.0))
;;; x47019 -&gt; (fn [] (normal 0.0 1.0))
;;; x47169 -&gt; (fn [] (normal 0.0 1.0))
;;; x47079 -&gt; (fn [] (normal 0.0 1.0))
;;; x47412 -&gt; (fn [] (normal 0.0 1.0))
;;; x47261 -&gt; (fn [] (normal 0.0 1.0))
;;; x47060 -&gt; (fn [] (normal 0.0 1.0))
;;; x47120 -&gt; (fn [] (normal 0.0 1.0))
;;; x47360 -&gt; (fn [] (normal 0.0 1.0))
;;; x47402 -&gt; (fn [] (normal 0.0 1.0))
;;; x47323 -&gt; (fn [] (normal 0.0 1.0))
;;; x47397 -&gt; (fn [] (normal 0.0 1.0))
;;; x47459 -&gt; (fn [] (normal 0.0 1.0))
;;; x47043 -&gt; (fn [] (normal 0.0 1.0))
;;; x47375 -&gt; (fn [] (normal 0.0 1.0))
;;; x47333 -&gt; (fn [] (normal 0.0 1.0))
;;; x47449 -&gt; (fn [] (normal 0.0 1.0))
;;; x47144 -&gt; (fn [] (normal 0.0 1.0))
;;; x47271 -&gt; (fn [] (normal 0.0 1.0))
;;; x47293 -&gt; (fn [] (normal 0.0 1.0))
;;; x47241 -&gt; (fn [] (normal 0.0 1.0))
;;; x47345 -&gt; (fn [] (normal 0.0 1.0))
;;; x47149 -&gt; (fn [] (normal 0.0 1.0))
;;; x47214 -&gt; (fn [] (normal 0.0 1.0))
;;; x47234 -&gt; (fn [] (normal 0.0 1.0))
;;; x47385 -&gt; (fn [] (normal 0.0 1.0))
;;; x47055 -&gt; (fn [] (normal 0.0 1.0))
;;; x47009 -&gt; (fn [] (normal 0.0 1.0))
;;; x47422 -&gt; (fn [] (normal 0.0 1.0))
;;; x47179 -&gt; (fn [] (normal 0.0 1.0))
;;; x47479 -&gt; (fn [x47298 x47219 x47019 x47390 x47159 x47108 x47417 x47048 x47209 x47055 x47224 x47266 x47072 x47149 x47313 x47338 x47084 x47286 x47204 x47464 x47407 x47031 x47303 x47234 x47144 x47422 x47091 x47345 x47370 x47427 x47380 x47024 x47132 x47014 x47169 x47120 x47293 x47036 x47229 x47009 x47127 x47079 x47459 x47375 x47214 x47323 x47350 x47256 x47328 x47189 x47096 x47276 x47174 x47308 x47154 x47043 x47184 x47103 x47365 x47449 x47246 x47164 x47199 x47261 x47385 x47355 x47397 x47469 x47454 x47318 x47412 x47271 x47115 x47360 x47442 x47139 x47179 x47067 x47060 x47194 x47437 x47333 x47251 x47241 x47432 x47402 x47281] (flip (nth (div 1.0 (add 1.0 (mat/exp (sub 0.0 (add (mmul [[x47189 x47194 x47199 x47204 x47209 x47214 x47219 x47224 x47229 x47234] [x47241 x47246 x47251 x47256 x47261 x47266 x47271 x47276 x47281 x47286] [x47293 x47298 x47303 x47308 x47313 x47318 x47323 x47328 x47333 x47338] [x47345 x47350 x47355 x47360 x47365 x47370 x47375 x47380 x47385 x47390] [x47397 x47402 x47407 x47412 x47417 x47422 x47427 x47432 x47437 x47442]] (mul (mat/ge (add (mmul [[x47019 x47024] [x47031 x47036] [x47043 x47048] [x47055 x47060] [x47067 x47072] [x47079 x47084] [x47091 x47096] [x47103 x47108] [x47115 x47120] [x47127 x47132]] [x47009 x47014]) [x47139 x47144 x47149 x47154 x47159 x47164 x47169 x47174 x47179 x47184]) 0.0) (add (mmul [[x47019 x47024] [x47031 x47036] [x47043 x47048] [x47055 x47060] [x47067 x47072] [x47079 x47084] [x47091 x47096] [x47103 x47108] [x47115 x47120] [x47127 x47132]] [x47009 x47014]) [x47139 x47144 x47149 x47154 x47159 x47164 x47169 x47174 x47179 x47184]))) [x47449 x47454 x47459 x47464 x47469]))))) 1)))
;;; 
;;; Observed values O:
;;; 
;;; number of arcs: 435
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; @@

;; @@
