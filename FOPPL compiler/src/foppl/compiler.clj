;; change the part of compute log_pdf(for continuous)/log_pmf(for discrete)
;; some problem in nth, python and tf index

(ns foppl.compiler
  (:require [foppl.desugar :refer :all]
            [foppl.core :as foppl :refer [foppl-query print-graph]] :reload
            [clojure.string :as str]
            [clojure.pprint :as ppt])
  (:use [anglican runtime emit]
        [clojure.inspector :include (atom?)]))

;;; vetices -> value
;;; p -> new node for the pdf
(def vertice-p (atom {}))
(defn update-v-p [k v]
  (swap! vertice-p assoc k v))

;; clojure list [a b c ...] to python list [a, b, c, ...]
;; input: clojure vector/list ...
;; output: python list string

(defn to-python-list [v]
  (let [s (str/join "," v)]
    (str/join ["[" s "]"])))

(to-python-list (vector 1 2 3))

(defn python-add [arg]
  (let [s "tf.add_n(["
        rest-arg (str/join "," arg)]
    (str/join [s, rest-arg, "])\n"])))
(python-add (vector 1 2 3))


;; evaluate expr
;; only translate the primitive functions and dist by now (with nested expr evaluated)
   ;;; basically take out the nested components as variables evaluated before
   ;;; input: expr
   ;;; output: [var-n var-s]
(defn tf-primitive [expr]
  (cond (seq? expr)
        (case (name (first expr))
          "not"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                less-n (gensym "x")
                less-s (str/join [fir-expr
                                  less-n " = tf.logical_not(" fir-n ")\n"])]
            (vector less-n less-s))

          "and"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = tf.logical_and(" fir-n ", " sec-n ")\n"])]
            (vector less-n less-s))
          "or"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = tf.logical_or(" fir-n ", " sec-n ")\n"])]
            (vector less-n less-s))

          "<"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = tf.less_equal(" fir-n ", " sec-n ")\n"])]
            (vector less-n less-s))
          "<="
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = tf.less(" fir-n ", " sec-n ")\n"])]
            (vector less-n less-s))
          ">"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = tf.greater(" fir-n ", " sec-n ")\n"])]
            (vector less-n less-s))
          ">="
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = tf.greater_equal(" fir-n ", " sec-n ")\n"])]
            (vector less-n less-s))

          ("+" "sum")
          (loop [arg-list []
                 add-string ""
                 rest-arg (rest expr)]
            ;;; traverse all the arg
            (if (empty? rest-arg)
              (let [add-n (gensym "x")    ;; attention, here the arglist without ","
                    add-s (str/join [add-string add-n " = " (python-add arg-list)])]
                  (vector add-n add-s))
              ;;; deep first
              (if (seq? (first rest-arg))
                  (let [temp (tf-primitive (first rest-arg))]
                    (recur (conj arg-list (first temp) )
                           (str/join [add-string (second temp)])
                           (rest rest-arg)))
                   (recur (conj arg-list (first rest-arg))
                          add-string
                          (rest rest-arg)))))
          "-"
          (if (empty? (rest (rest expr)))
            ;; eg. (- 1)
            (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                  less-n (gensym "x")
                  less-s (str/join [fir-expr
                                    less-n " = tf.negative(" fir-n ")\n"])]
              ;(prn "entry ")
              (vector less-n less-s))
            ;; eg. (- 1 2)
            (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                  [sec-n sec-expr] (tf-primitive (nth expr 2))
                  less-n (gensym "x")
                  less-s (str/join [fir-expr sec-expr
                                    less-n " = tf.subtract(" fir-n ", " sec-n ")\n"])]
              (vector less-n less-s)))

          "*"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = tf.multiply(" fir-n ", " sec-n ")\n"])]
            (vector less-n less-s))
          "/"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = tf.divide(" fir-n ", " sec-n ")\n"])]
            (vector less-n less-s))


          "exp"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                less-n (gensym "x")
                less-s (str/join [fir-expr
                                  less-n " = tf.exp (" fir-n ")\n"])]
            (vector less-n less-s))


          ;;; continous dist
          "normal"
           (let [[mu mu-string] (tf-primitive (first (rest expr)))
                 [std std-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "x")
                 dist-string (str/join [mu-string std-string
                                        dist-n " = Normal(mu=" mu ", sigma=" std ")\n"])]
             (vector dist-n dist-string))

          "bernoulli"
           (let [[p p-string] (tf-primitive (first (rest expr)))
                 dist-n (gensym "x")
                 dist-string (str/join [p-string
                                        dist-n " = Bernoulli(probs=" p ")\n"])]
             (vector dist-n dist-string))
          "binomial"
           (let [[n n-string] (tf-primitive (first (rest expr)))
                 [p p-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "x")
                 dist-string (str/join [n-string p-string
                                        dist-n " = Binomial(total_count=" n ", probs=" p ")\n"])]
             (vector dist-n dist-string))
          "beta"
           (let [[a a-string] (tf-primitive (first (rest expr)))
                 [b b-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "x")
                 dist-string (str/join [a-string b-string
                                        dist-n " = Beta(" a ", " b ")\n"])]
             (vector dist-n dist-string))
          "gamma"
           (let [[a a-string] (tf-primitive (first (rest expr)))
                 [b b-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "x")
                 dist-string (str/join [a-string b-string
                                        dist-n " = Gamma(" a ", " b ")\n"])]
             (vector dist-n dist-string))
          "uniform-continuous"
           (let [[low low-string] (tf-primitive (first (rest expr)))
                 [high high-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "x")
                 dist-string (str/join [low-string high-string
                                        dist-n " = Uniform(low=" low ", high=" high ")\n"])]
             (vector dist-n dist-string))

          "exponential"
           (let [[l l-string] (tf-primitive (first (rest expr)))
                 dist-n (gensym "x")
                 dist-string (str/join [l-string
                                        dist-n " = Exponential(rate=" l ")\n"])]
             (vector dist-n dist-string))

          "discrete"  ;; translate to categorical in tf
           (let [[p p-string] (tf-primitive (first (rest expr)))
                 dist-n (gensym "x")
                 dist-string (str/join [p-string
                                        dist-n " = Categorical(p=" p ")\n"])]
             (vector dist-n dist-string))
          ;;; discrete
          "poisson"
           (let [[l l-string] (tf-primitive (first (rest expr)))
                 dist-n (gensym "x")
                 dist-string (str/join [l-string
                                        dist-n " = Poisson(" l ")\n"])]
             (vector dist-n dist-string))


          "if" ;; if in sample (if cond expr1 expr2)
               ;; if in obs (if cond expr1), only when cond is satisfied, calculate pdf according to expr1
          (case (count expr)
                4
               (let [[e1-n e1-s] (tf-primitive (nth expr 1))
                     [e2-n e2-s] (tf-primitive (nth expr 2))
                     [e3-n e3-s] (tf-primitive (nth expr 3))
                     if-n (gensym "x")
                     if-s (str/join [e1-s e2-s e3-s
                                     if-n " = tf.cond(" e1-n ", lambda: " e2-n ", lambda: " e3-n ")\n"])]
                       (vector if-n if-s))

                3
               (let [[e1-n e1-s] (tf-primitive (nth expr 1))
                     [e2-n e2-s] (tf-primitive (nth expr 2))
                     [e3-n e3-s] (tf-primitive (nth expr 2))
                     if-n (gensym "x")
                     if-s (str/join [e1-s e2-s e3-s
                                     if-n " = tf.cond(" e1-n ", lambda: " e2-n ", lambda: " e3-n ")\n"])]
                       (vector if-n if-s))

                "default")

          "nth"
          (let [[num-n num-s] (tf-primitive (nth expr 2))
                [elem-n elem-s] (tf-primitive (nth expr 1))
                r-n (gensym "x")
                r-s (str/join [num-s elem-s
                               r-n " = tf.gather(" elem-n ", tf.to_int32(" num-n "))\n"])]
            (vector r-n r-s))

           (prn "No match case!" expr))


        ;(or (vector? expr) (list? expr))
        ;; ATTENTION!!! (list 1 2) is a seq, would go into previous one
        ;; if returning E is a list, should be careful
        (vector? expr)
        (if (every? atom? expr)
          (let [v-n (gensym "x")
                v-s (str/join [v-n " = " (to-python-list expr) "\n"])]
            ;(prn "all atom")
            (vector v-n v-s))
          (loop [arg-list []
                 add-string ""
                 rest-arg expr]
            ;;; traverse all the arg
            (if (empty? rest-arg)
              (let [add-n (gensym "x")    ;; attention, here the arglist without ","
                    add-s (str/join [add-string add-n " = " (to-python-list arg-list) "\n"])]
                  (vector add-n add-s))
              ;;; deep first
              (if (not (atom? (first rest-arg)))  ;; previous seq? could not distinguish vector inside vector
                  (let [temp (tf-primitive (first rest-arg))]
                    (recur (conj arg-list (first temp) )
                           (str/join [add-string (second temp)])
                           (rest rest-arg)))
                   (recur (conj arg-list (first rest-arg))
                          add-string
                          (rest rest-arg))))))

        (number? expr)
        (let [e-n (gensym "c")
              e-s (str/join[e-n "= tf.constant(" expr")\n"])]
          (vector e-n e-s))

        :else (vector expr "")))

;;; combine tf-var-expr and tf-var-declare
(defn tf-var-declare [foppl-query]
  (let [P (nth (first foppl-query) 2)
        O (nth (first foppl-query) 3)]
    (loop [var-list (foppl/topo-sort foppl-query)
           str-prog ""]
      ;(prn var-list)
      (if (empty? var-list)
        str-prog
        (let [var-n (first var-list)
              expr (:body (get P var-n))
              [var-e var-s]  (tf-primitive expr)] ;final return would always be the tf dist obj
          (if (contains? O var-n)
            ;; observe vertices
            (let [o-value (get O var-n)
                  [o-n o-s] (tf-primitive o-value)
                  var-string (str/join [str-prog var-s o-s
                                        var-n " = " o-n " \n"])
                  pdf-n (gensym "p")
                  pdf-s (str/join [pdf-n " = "  var-e ".log_pdf( " var-n ") if "  var-e
                                         ".is_continuous else "  var-e ".log_pmf( " var-n ") #obs, log likelihood\n"])]
                 (update-v-p var-n pdf-n)  ;; update the atomn
                 (recur (rest var-list)
                        (str/join [var-string pdf-s])))
            ;; sample vertices
            (let [var-string (str/join [str-prog var-s
                                         var-n " = tf.Variable( " var-e ".sample())   #sample\n"])
                  pdf-n (gensym "p")
                  pdf-s (str/join [pdf-n " = " var-e ".log_pdf( " var-n ") if " var-e
                                     ".is_continuous else " var-e ".log_pmf( " var-n ")   #prior\n"])]
                (update-v-p var-n pdf-n)  ;;update the atom
                (recur (rest var-list)
                        (str/join [var-string pdf-s])))))))))


;;; add all the log-pdf of likelihood together
(defn tf-joint-log-pdf [foppl-query]
   (let [add-n (gensym "p")
         pdf-n (vals @vertice-p)
         add-s (str/join [add-n " = " (python-add pdf-n)])]
     (vector add-n add-s)))
;; (tf-joint-log-pdf foppl-src00)


;;;session and run part

;; init order, because of the tf.Variable in sample
;; foppl/topo-sort returns the evaluation order
(defn init-order [foppl-query]
  (loop [rest-v (foppl/topo-sort foppl-query)
         init-s ""]
    (if (empty? rest-v)
      init-s
      (let [v-n (first rest-v)
            O (nth (first foppl-query) 3)]
        (if (contains? O v-n)
          (recur (rest rest-v) init-s)
          ;; only the vertices in sample statements are type variable
          (recur (rest rest-v) (str/join [init-s "sess.run(" v-n ".initializer)\n"])))))))

; (init-order foppl-src1)


;; add heading, eg "import expr as e"
(defn add-heading [expr e]
  (str/join ["import " expr " as " e "\n"
             "from tensorflow.contrib.distributions import * \n"]))

;(add-heading "tensorflow" "tf")

(defn sess-declare [session sess]
  (str/join [sess " = " session "\n"]))

;(sess-declare "tf.Session()" "sess")

(defn sess-close []
  (let [graph-dir (str/join ["./Graph_Output/" (gensym "g")])]
    ;(prn graph-dir)
    (str/join ["writer = tf.summary.FileWriter( '"  graph-dir  "', sess.graph)\n"
               "sess.close()"])))
;(sess-close)


;;; deal with the E
;; output [s1, s2] two string, one declare, one init and run
(defn eval-E [foppl-query]
  (let [E (second foppl-query)
        E-expr (:body E)
        G (first foppl-query)]
    ;(prn E-expr)
    (cond (not (atom? E-expr))   ;; returning sequence
          (let [[s1-n s1] (tf-primitive E-expr)
                s2 (str/join ["print(sess.run(" s1-n "))\n"])]
            ;(prn s1-n s1)
            (vector s1 s2))

          (contains? (first G) E-expr)  ;; random varibales
          (let [s1 ""
                s2 (str/join["print(sess.run(" E-expr "))\n"])]
            (vector s1 s2))

          (atom? E-expr)  ;;number, string
          (let [s1 ""
                s2 (str/join["print(" E-expr ")\n"])]
            (vector s1 s2))

          :else "Not Match!")))

;; (eval-E test-src)


;;; __main
(defn compile-query [foppl-query]
  (reset! vertice-p {})
  (let [heading (add-heading "tensorflow" "tf")
        declare-s (tf-var-declare foppl-query)
        [pdf-n pdf-s] (tf-joint-log-pdf foppl-query)
        [declare-E run-E] (eval-E foppl-query)]
    (str/join [heading
               declare-s
               pdf-s
               "# return E from the model\n"
               declare-E
               "\n"
               (sess-declare "tf.Session()" "sess")
               (init-order foppl-query)
               "sess.run("pdf-n")\n"
               "# printing E: \n"
               run-E
               (sess-close)])))

