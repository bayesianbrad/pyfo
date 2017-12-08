(ns foppl.desugar)


(defn desugar-let-bindings
  "Desugar a multiple-binding let into nested single-binding lets"
  [expr]
  (if-not (seq? expr)
    expr
    (if (= (-> expr first name) "let")
      (let [[_ bindings body] expr]
        (let [[k v & others] bindings
              new-bindings (vector k (desugar-let-bindings v))
              body (desugar-let-bindings body)]
          (if others
            (list 'let new-bindings (desugar-let-bindings (list 'let (into [] others) body)))
            (list 'let new-bindings body))))
      (cons (first expr) (doall (map desugar-let-bindings (rest expr)))))))

(defn desugar-let-body
  "Desugar a multiple-statement let body into a multiple-binding let"
  [expr]
  (if-not (seq? expr)
    expr
    (if (and (= (-> expr first name) "let")
             (> (count expr) 3))
      (let [[_ bindings & statements] expr
            statements (reverse statements)
            body (first statements)
            new-bindings (loop [statements (reverse (rest statements))
                                bindings bindings]
                           (if (> (count statements) 0)
                             (recur (rest statements)
                                    (concat bindings [(gensym "let-body") (first statements)]))
                             (doall bindings)))]
        (list 'let (into [] new-bindings) body))
      (cons (first expr) (doall (map desugar-let-body (rest expr)))))))

(defn desugar-loop
  "Desguar a loop into a multiple-binding let"
  [expr]
  (if-not (seq? expr)
    expr
    (case (-> expr first name)
      "loop" (let [[_ iters init f & args] expr
                   iters (eval iters)
                   args (doall (map desugar-loop args))]
               (assert (and (int iters) (> iters 0)))
               (loop [i 0
                      bindings []
                      state (desugar-loop init)]
                 (if (= i iters)
                   (list 'let (into [] bindings) state)
                   (let [next-sym (gensym "loop")
                         next-value (concat [f i state] args)]
                     ;;(println "Next value:" next-value)
                     (recur (+ i 1)
                            (concat bindings [next-sym next-value])
                            next-sym)))))
       "let" (let [[_ bindings & statements] expr
                   new-bindings (loop [bindings bindings
                                       new-bindings []]
                                  (if (= (count bindings) 0)
                                    new-bindings
                                    (let [lhs (first bindings)
                                          rhs (desugar-loop (second bindings))]
                                      (recur (rest (rest bindings))
                                             (conj (conj new-bindings lhs) rhs)))))]
               (concat (list 'let (into [] new-bindings)) (doall (map desugar-loop statements))))
      (cons (first expr) (doall (map desugar-loop (rest expr)))))))

(defn desugar
  "Desugar an expression"
  [expr]
  ;(let [no-loops (into [] (desugar-loop expr))
  ;      desugared (desugar-let-bindings (desugar-let-body no-loops))]
  ;  desugared))
  (-> expr desugar-loop desugar-let-body desugar-let-bindings))
