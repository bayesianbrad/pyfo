(ns foppl.core
  (:require [clojure.set :refer [union intersection]]
            [foppl.desugar :refer [desugar]]
            [anglican.runtime :refer [distribution]]
            :reload))

(defrecord Variable [name]
  Object (toString [self] (str (:name self))))

(defrecord Expression [expr]
  Object (toString [self] (str (:expr self))))

(defrecord Trace [rdb ordering logprob])

(defn variable? [v] (instance? Variable v))
(defn expression? [e] (instance? Expression e))
(defn static-determinable? [e] (not (or (variable? e) (expression? e))))

(defn unbox [e]
  (clojure.walk/postwalk #(if (expression? %) (:expr %) %) e))

(defn unquote-variables [expr]
  (clojure.walk/postwalk #(if (variable? %) (:name %) %) expr))

;; Simple thin wrapper for compiled clojure functions
(defrecord ClojureFunction [args body fn]
  Object (toString [self] (str (list 'fn args body))))

(defn wrap-fn [args body]
  (let [args (into [] args)]
    (ClojureFunction. args body
                      (let [argsname (gensym 'args)]
                        (eval `(fn [& ~argsname] (let [~args ~argsname] ~body)))))))
                               ;;(eval (list 'fn ['& argsname] (list let [args body))))))


(def G_empty [#{} #{} {} {}])

;; primitive procedures which are safe to evaluate on boxed expressions
(def whitelist #{'conj 'nth 'first 'second 'rest 'last 'vector 'list 'get})

(defn merge-graph [G1 G2]
  (let [[V1 A1 P1 O1] G1
        [V2 A2 P2 O2] G2]
    [(union V1 V2)
     (union A1 A2)
     (merge P1 P2)
     (merge O1 O2)]))

(defn substitute [e x E]
  ;; TODO: will this fail on more esoteric data structures?
  ;;       Possibly would be better to implement using `clojure.walk`
  (cond
    (symbol? e) (if (= x e) E e)
    (vector? e) (mapv #(substitute % x E) e)
    (seq? e) (if (= (-> e first name) "let")
               (let [[_ bindings body] e
                     [k value] bindings
                     new-bindings (vector k (substitute value x E))]
                 (if (= k x)
                   (list 'let new-bindings body)
                   (list 'let new-bindings (substitute body x E))))
               (let [func (first e)]
                 ;(println "PROCESSING FUNCTION CALL" func "IN" e)
                 (loop [new-args (list)
                        old-args (rest e)]
                   ;(println "... new" new-args)
                   ;(println "... old" old-args " -- empty?" (empty? old-args))
                   (if (empty? old-args)
                     (cons func (reverse new-args))
                     (let [new-arg (substitute (first old-args) x E)]
                      ; (println "new-arg:" new-arg)
                       (recur (cons new-arg new-args) (rest old-args)))))))
    :else e))

(defn debug-substitute [e x E]
  (println (str "[PRE] substitute symbol " x " with expression " E " in original expression\n   " e))
  (let [result (substitute e x E)]
    (println "[POST]" result)
    result))

(defn freevars [expr]
  (let [fv (atom #{})]
    (clojure.walk/postwalk #(if (variable? %) (swap! fv conj %) %) expr)
    @fv))

(defn eval-or-box
  "Attempt to evaluate an expression, and box the expression if it fails"
  [expr]
  (let [do-i-eval? (or (empty? (freevars expr))
                       (contains? whitelist (first expr)))]
    (if do-i-eval?
      (let [result (try (eval expr)
                        (catch Exception e
                          (Expression. expr)))]
        (cond (seq? result) (cons 'list result)
              (satisfies? distribution result) (Expression. expr)
              :else result))
      (Expression. expr))))

(defn partial-eval
  "Partial evaluation"
  [expr]
  (let [is-fn-call (seq? expr)
        result (if is-fn-call (eval-or-box expr) expr)]
    ;;(println (str "[partial in] " expr "\n[partial out] " result))
    result))

(defn- include-expr-in-phi [phi e]
  (if (= phi true) e (list 'and e phi)))

(defn compile-graph [expr phi rho]
  ;; (println (str "Compiling expression: " expr))
  (cond (seq? expr)
        (case (name (first expr))
          ; let
          "let" (let [[[x e1] e2] (rest expr)
                      [G1 E1] (compile-graph e1 phi rho)
                      new-body (substitute e2 x E1)
                      [G2 E2] (compile-graph new-body phi rho)]
                  [(merge-graph G1 G2) E2])

          ; if
          "if" (let [[e1 e2 e3] (rest expr)
                     [G1 E1] (compile-graph e1 phi rho)
                     [G2 E2] (compile-graph e2 (include-expr-in-phi phi E1) rho)
                     [G3 E3] (compile-graph e3 (include-expr-in-phi phi (list 'not E1)) rho)]
                 [(merge-graph (merge-graph G1 G2) G3)
                  (partial-eval (list 'if E1 E2 E3))])

          ; sample
          "sample" (let [[[V A P O] E] (compile-graph (second expr) phi rho)
                         Y (freevars E) ; NOTE: removed intersection with V
                         x (Variable. (gensym "x"))
                         ;; Do we want to return a scoring expression, or a dist?
                         ;; E' {:args (into [] Y) :dist E}
                         E' [(into [] Y) E]
                         arcs (into #{} (for [y Y] [y x]))]
                     [[(union V #{x}) (union A arcs)
                       (assoc P x E') O]
                      x])

          ; observe
          "observe" (let [[e1 e2] (rest expr)
                          [G1 y] (compile-graph (list 'sample e1) phi rho)
                          [G2 E2] (compile-graph e2 phi rho)
                          [V A P O] (merge-graph G1 G2)]
                      [[V A P (assoc O y {:value E2 :cond [(freevars phi) phi]})] E2])

          ; function application
          (if (contains? rho (first expr))
            ; user-defined procedure
            (let [func (first expr)
                  L (map #(compile-graph % phi rho) (rest expr))
                  Es (map second L)
                  [_ _ args e] (get rho func)]
              ;; (println (str "[DEBUG] applying user-defined procedure" func "to" Es))
              (loop [e e
                     args args
                     Es Es]
                (if (empty? args)
                  (let [[G E] (compile-graph e phi rho)]
                    [(reduce merge-graph G (map first L))  E])
                  (recur (substitute e (first args) (first Es))
                         (rest args)
                         (rest Es)))))
            ; primitive procedure
            (let [func (first expr)
                  L (map #(compile-graph % phi rho) (rest expr))
                  G (reduce merge-graph G_empty (map first L))
                  Es (map partial-eval (map second L))
                  E (cons func Es)]
              [G (partial-eval E)])))
        :else [G_empty expr]))

(defn freevars-match-arcs?
  "Debugging: for any graph, the free variables in the expressions for P
  should match the parent set as defined by the arcs in A.
  Additionally, the keys of P should be exactly the entries in V."
  [G]
  (let [[V A P O] G
        nodes (into {} (map #(vector (second %) (atom #{})) A))]
    (assert (= V (into #{} (keys P))))
    (doall (for [[w v] A] (swap! (get nodes v) conj w)))
    (let [nodes (into {} (for [[a b] nodes] [a @b]))]
      ;;(println "matches?"
      ;;         (for [[v pv] P] [v (= (freevars pv) (conj (get nodes v #{}) v))]))
      (reduce #(and %1 %2) true
              (for [[v pv] P]
                ;; The following depends on how we defined the score expression
                (= (freevars pv) (get nodes v #{})))))))
                ;;(= (freevars pv) (conj (get nodes v #{}) v)))))))

(defn get-latents [foppl-model]
  (let [[V _ _ O] (first foppl-model)]
    (clojure.set/difference V (keys O))))

(defn compile-graph-expressions [model]
  (let [[[V A P O] E] model
        E' (wrap-fn (get-latents model) E)
        P' (into {} (for [[k v] P] [k (wrap-fn (first v) (second v))]))
        O' (into {} (for [[k v] O] [k (assoc v :cond (apply wrap-fn (:cond v)))]))]
    `((~V ~A ~P' ~O') ~E')))

(defmacro foppl-query [& body]
  (loop [expr# (first body)
         body (rest body)
         rho# {}]
    ;;(println "compiling expression:" expr#)
    ;;(println "Function definitions:" rho#)
    (if (empty? body)
      `(let [[G# E#] (compile-graph '~(desugar expr#) true ~rho#)]
         ;;[G# E#])
         (assert (freevars-match-arcs? G#))
         (compile-graph-expressions (unquote-variables (unbox [G# E#]))))
      (let [f# (second expr#)]
        (assert (= (first expr#) 'defn))
        (recur (first body) (rest body) (assoc rho# `'~f# `'~(desugar expr#)))))))

;;; Some useful utility functions

(defn print-graph
  "pretty-print a compiled graph"
  [G]
  (let [[V A P O] G]
    (println "Vertices V:" V)
    (println "\nArcs A:" A)
    (println "\nConditional densities P:")
    (println (clojure.string/join "\n" (for [[k v] P] (str k " -> " v)))) ;;(list 'fn (:args v) (:dist v))))))
    (println "\nObserved values O:")
    (println (clojure.string/join "\n" (for [[k v] O] (str k " -> " (:value v) "; " (:cond v)))))))

(defn get-edges "return all edges as parent->{children} map" [foppl-model]
  (let [raw-edges (second (first foppl-model))]
    (apply merge-with clojure.set/union (map (fn [[u v]] {u #{v}}) raw-edges))))

(defn get-parents "return map of parent sets child->{parents}" [foppl-model]
  (let [raw-edges (second (first foppl-model))]
    (apply merge-with clojure.set/union (map (fn [[v u]] {u #{v}}) raw-edges))))

(defn get-nodes [foppl-model]
  (first (first foppl-model)))

(defn visit [n edges L unmarked]
  ;;(println "[debug] (visit" n edges L unmarked ")")
  ;;(assert (contains? unmarked n))
  (if (contains? unmarked n)
    (loop [L L
           unmarked unmarked
           children (into '() (get edges n #{}))]
      (if (empty? children)
        (do
          ;;(println "insert" n "at front of" L)
          [(conj L n) (disj unmarked n)])
        (let [m (first children)
              ;;_ (println "visiting" m "from" n)
              [L unmarked] (visit m edges L (disj unmarked n))]
          (recur L unmarked (rest children)))))
    [L unmarked]))

(defn topo-sort [foppl-model]
  (let [edges (get-edges foppl-model)]
    (loop [L '()
           unmarked (get-nodes foppl-model)]
      (if (empty? unmarked)
        L
        (let [n (first unmarked)
              [L unmarked] (visit n edges L unmarked)]
          ;;(println n "->" L)
          (recur L unmarked))))))

(defn get-dist-at-addr
  "Helper function for evaluating probability expressions"
  [trace P addr]
  (let [func (P addr)
        rdb (:rdb trace)]
    (apply (:fn func) (map rdb (:args func)))))

(defn compute-logprob
  "Given a model and a trace, compute its log probability"
  [foppl-model trace]
  (let [[[V A P O] E] foppl-model]
    (loop [logprob 0.0
           nodes (:ordering trace)]
      (if (empty? nodes)
        logprob
        (let [address (first nodes)
              dist (get-dist-at-addr trace P address)]
          (if (contains? O address)
            ;; Check if we include this observe in this trace
            (let [include-fn (:cond (O address))
                  include-args (mapv #(get (:rdb trace) %) (:args include-fn))
                  include? (apply (:fn include-fn) include-args)]
              (if include?
                (recur (+ logprob
                          (anglican.runtime/observe* dist (:value (O address))))
                       (rest nodes))
                ;; if not, don't increment logprob
                (recur logprob (rest nodes))))
            (recur (+ logprob
                      (anglican.runtime/observe* dist (get-in trace [:rdb address])))
                   (rest nodes))))))))

(defn draw-from-prior
  "Return a trace object, corrsponding to a sample from the 'prior'
  distribution (i.e. ignoring observe statements)"
  [foppl-model]
  (let [[[V A P O] E] foppl-model
        ordering (topo-sort foppl-model)
        latents (clojure.set/difference V (keys O))
        parents (get-parents foppl-model)]
    (loop [trace (Trace. {} ordering nil)
           nodes ordering]
      (if (empty? nodes)
        (assoc trace :logprob (compute-logprob foppl-model trace))
        (if (contains? O (first nodes))
          (recur trace (rest nodes))
          (let [p (get P (first nodes))
                args (mapv #(get (:rdb trace) %) (:args p))]
            (let [value (anglican.runtime/sample* (apply (:fn p) args))
                  trace (assoc-in trace [:rdb (first nodes)] value)]
              (recur trace (rest nodes)))))))))

(defn get-output
  "Given a model and a trace, return the program output"
  [foppl-model trace]
  (let [E (second foppl-model)
        rdb (:rdb trace)]
    (apply (:fn E) (map #(rdb %) (:args E)))))
