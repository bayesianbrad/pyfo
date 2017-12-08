;; gorilla-repl.fileformat = 1

;; **
;;; # Beta-Binomial example
;;; 
;; **

;; @@
(ns simple-beta-flip
  (:require [gorilla-plot.core :as plot]
            [foppl.desugar :refer :all]
            [foppl.core :as foppl :refer [foppl-query print-graph]] :reload)
  (:use [anglican runtime]))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; **
;;; ## Calling `foppl-query` returns a tuple with two entries.
;;; 
;;; The first is a graph representation, and the second is an expression for the return value.
;; **

;; @@
(def beta-flip
  (foppl-query
    (let [p (sample (beta (sample (exponential 1.0)) 1))
          d (bernoulli p)]
      (observe d 1)
      (observe d 1)
      (observe d 0)
      (observe d 1)
      p)))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;simple-beta-flip/beta-flip</span>","value":"#'simple-beta-flip/beta-flip"}
;; <=

;; @@
(print-graph (first beta-flip))
;; @@
;; ->
;;; Vertices V: #{x65926 x65927 x65925 x65923 x65924 x65922}
;;; 
;;; Arcs A: #{[x65923 x65925] [x65922 x65923] [x65923 x65927] [x65923 x65926] [x65923 x65924]}
;;; 
;;; Conditional densities P:
;;; x65922 -&gt; (fn [] (exponential 1.0))
;;; x65923 -&gt; (fn [x65922] (beta x65922 1))
;;; x65924 -&gt; (fn [x65923] (bernoulli x65923))
;;; x65925 -&gt; (fn [x65923] (bernoulli x65923))
;;; x65926 -&gt; (fn [x65923] (bernoulli x65923))
;;; x65927 -&gt; (fn [x65923] (bernoulli x65923))
;;; 
;;; Observed values O:
;;; x65924 -&gt; 1; (fn [] true)
;;; x65925 -&gt; 1; (fn [] true)
;;; x65926 -&gt; 0; (fn [] true)
;;; x65927 -&gt; 1; (fn [] true)
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; @@
(:body (second beta-flip))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-symbol'>x65923</span>","value":"x65923"}
;; <=

;; **
;;; Calling `foppl/init-latents` will draw a sample from the prior distribution, by performing a topological sort on the graph, and sampling from each conditional distribution in turn.
;;; 
;;; Actual inference methods that take advantage of the graph haven't been implemented yet.
;; **

;; @@
(foppl/init-latents beta-flip)
;; @@
;; =>
;;; {"type":"list-like","open":"<span class='clj-map'>{</span>","close":"<span class='clj-map'>}</span>","separator":", ","items":[{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x65922</span>","value":"x65922"},{"type":"html","content":"<span class='clj-double'>1.4709149870427358</span>","value":"1.4709149870427358"}],"value":"[x65922 1.4709149870427358]"},{"type":"list-like","open":"","close":"","separator":" ","items":[{"type":"html","content":"<span class='clj-symbol'>x65923</span>","value":"x65923"},{"type":"html","content":"<span class='clj-double'>0.0469242989878251</span>","value":"0.0469242989878251"}],"value":"[x65923 0.0469242989878251]"}],"value":"{x65922 1.4709149870427358, x65923 0.0469242989878251}"}
;; <=

;; **
;;; ### A different program (previously, this did not compile correctly)
;; **

;; @@
(def problem-child
  (foppl-query
    (if (< 1.5 (+ 1 (sample (flip 0.5))))
      3
      (let [_ (observe (normal 0 1) 2)]
        4))))
;; @@
;; =>
;;; {"type":"html","content":"<span class='clj-var'>#&#x27;simple-beta-flip/problem-child</span>","value":"#'simple-beta-flip/problem-child"}
;; <=

;; @@
(print-graph (first problem-child))
;; @@
;; ->
;;; Vertices V: #{x66007 x66010}
;;; 
;;; Arcs A: #{}
;;; 
;;; Conditional densities P:
;;; x66007 -&gt; (fn [] (flip 0.5))
;;; x66010 -&gt; (fn [] (normal 0 1))
;;; 
;;; Observed values O:
;;; x66010 -&gt; 2; (fn [x66007] (not (&lt; 1.5 (+ 1 x66007))))
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; @@
(println "return value" (:body (second problem-child)))
;; @@
;; ->
;;; return value (if (&lt; 1.5 (+ 1 x66007)) 3 4)
;;; 
;; <-
;; =>
;;; {"type":"html","content":"<span class='clj-nil'>nil</span>","value":"nil"}
;; <=

;; @@

;; @@
