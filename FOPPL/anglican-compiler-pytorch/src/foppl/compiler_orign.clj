;; origin compiler, RV generates samples

(ns foppl.compiler-origin
  (:require ;[foppl.desugar :refer :all]
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
;;   (let [s "tf.add_n(["
;;         rest-arg (str/join "," arg)]
;;     (str/join [s, rest-arg, "])\n"])))
  (let [add-str (str/join " + " arg)]
    (str/join [" " add-str " "])))
(python-add (vector 1 2 3))


(declare convert-dist)
(declare convert-primitive-opt)
;; evaluate expr
;; only translate the primitive functions and dist by now (with nested expr evaluated)
   ;;; basically take out the nested components as variables evaluated before
   ;;; input: expr
   ;;; output: [var-n var-s]
(defn tf-primitive [expr]
  (cond (seq? expr)
        (case (name (first expr))
          ;; primitive operation
          ("not" "and" "or" ">" ">=" "<" "<="
           "+" "sum" "-" "*" "/" "exp" "nth")
          (convert-primitive-opt expr)

          ;;; continous dist
          ("normal" "mvn" "discrete")
          (convert-dist expr)

          "if" ;; if in sample (if cond expr1 expr2)
               ;; if in obs (if cond expr1), only when cond is satisfied, calculate pdf according to expr1
          (case (count expr)
                4
               (let [[e1-n e1-s] (tf-primitive (nth expr 1))
                     [e2-n e2-s] (tf-primitive (nth expr 2))
                     [e3-n e3-s] (tf-primitive (nth expr 3))
                     if-n (gensym "x")
                     if-s (str/join [e1-s e2-s e3-s
                                     "if " e1-n " :\n"
                                     "\t " e2-n "\n"
                                     "else:"
                                     "\t " e3-n "\n"])]
                       (vector if-n if-s))

                3
               (let [[e1-n e1-s] (tf-primitive (nth expr 1))
                     [e2-n e2-s] (tf-primitive (nth expr 2))
                     ; [e3-n e3-s] (tf-primitive (nth expr 2))
                     if-n (gensym "x")
                     if-s (str/join [e1-s e2-s ;e3-s
                                     "if " e1-n " :\n"
                                     "\t " e2-n "\n"])]
                       (vector if-n if-s))
                "default")

           (prn "No match case!" expr)
          )

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

;;         (number? expr)
;;         (let [e-n (gensym "c")
;;               e-s (str/join[e-n "= VariableCast(" expr")\n"])]
;;           (vector e-n e-s))

        :else (vector expr "")))


;; (tf-primitive '(if (> x22867 0) (normal 1 1)))

(defn convert-primitive-opt [expr]
  (case (name (first expr))
          "not"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                less-n (gensym "x")
                less-s (str/join [fir-expr
                                  less-n " = not " fir-n "\n"])]
            (vector less-n less-s))

          "and"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " =" fir-n " and " sec-n "\n"])]
            (vector less-n less-s))
          "or"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = " fir-n " or " sec-n "\n"])]
            (vector less-n less-s))

          ">"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = " fir-n " > " sec-n "\n"])]
            (vector less-n less-s))
          ">="
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = " fir-n " >= " sec-n "\n"])]
            (vector less-n less-s))
          "<"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = " fir-n " < " sec-n "\n"])]
            (vector less-n less-s))
          "<="
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = " fir-n " <= " sec-n "\n"])]
            (vector less-n less-s))


          ("+" "sum")
          (loop [arg-list []
                 add-string ""
                 rest-arg (rest expr)]
            ;;; traverse all the arg
            (if (empty? rest-arg)
              (let [add-n (gensym "x")    ;; attention, here the arglist without ","
                    add-s (str/join [add-string add-n " = " (python-add arg-list) " \n"])]
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
                                    less-n " = - " fir-n "\n"])]
              ;(prn "entry ")
              (vector less-n less-s))
            ;; eg. (- 1 2)
            (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                  [sec-n sec-expr] (tf-primitive (nth expr 2))
                  less-n (gensym "x")
                  less-s (str/join [fir-expr sec-expr
                                    less-n " = " fir-n " - " sec-n "\n"])]
              (vector less-n less-s)))

          "*"
          ;; need to refine, matrix mul
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = " fir-n " * " sec-n "\n"])]
            (vector less-n less-s))
          "/"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = " fir-n "/ " sec-n "\n"])]
            (vector less-n less-s))


          "exp"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                less-n (gensym "x")
                less-s (str/join [fir-expr
                                  less-n " = torch.exp (" fir-n ")\n"])]
            (vector less-n less-s))

          "nth"
          (let [[num-n num-s] (tf-primitive (nth expr 2))
                [elem-n elem-s] (tf-primitive (nth expr 1))
                r-n (gensym "x")
                r-s (str/join [num-s elem-s
                               r-n " = " elem-n "[ int(" num-n ")]\n"])]
            (vector r-n r-s))
    ))


(defn convert-dist [expr]
  (case (name (first expr))
          "normal"
           (let [[mu mu-string] (tf-primitive (first (rest expr)))
                 [std std-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "x")
                 dist-string (str/join [mu-string std-string
                                        dist-n " = Normal(mean=" mu ", std=" std ")\n"])]
             (vector dist-n dist-string))

          "mvn"
           (let [[mu mu-string] (tf-primitive (first (rest expr)))
                 [cov cov-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "x")
                 dist-string (str/join [mu-string cov-string
                                        dist-n " = MultivariateNormal(mean=" mu ", cov=" cov ")\n"])]
             (vector dist-n dist-string))

;;           "bernoulli"
;;            (let [[p p-string] (tf-primitive (first (rest expr)))
;;                  dist-n (gensym "x")
;;                  dist-string (str/join [p-string
;;                                         dist-n " = Bernoulli(probs=" p ")\n"])]
;;              (vector dist-n dist-string))
;;           "binomial"
;;            (let [[n n-string] (tf-primitive (first (rest expr)))
;;                  [p p-string] (tf-primitive (second (rest expr)))
;;                  dist-n (gensym "x")
;;                  dist-string (str/join [n-string p-string
;;                                         dist-n " = Binomial(total_count=" n ", probs=" p ")\n"])]
;;              (vector dist-n dist-string))
;;           "beta"
;;            (let [[a a-string] (tf-primitive (first (rest expr)))
;;                  [b b-string] (tf-primitive (second (rest expr)))
;;                  dist-n (gensym "x")
;;                  dist-string (str/join [a-string b-string
;;                                         dist-n " = Beta(" a ", " b ")\n"])]
;;              (vector dist-n dist-string))
;;           "gamma"
;;            (let [[a a-string] (tf-primitive (first (rest expr)))
;;                  [b b-string] (tf-primitive (second (rest expr)))
;;                  dist-n (gensym "x")
;;                  dist-string (str/join [a-string b-string
;;                                         dist-n " = Gamma(" a ", " b ")\n"])]
;;              (vector dist-n dist-string))
;;           "uniform-continuous"
;;            (let [[low low-string] (tf-primitive (first (rest expr)))
;;                  [high high-string] (tf-primitive (second (rest expr)))
;;                  dist-n (gensym "x")
;;                  dist-string (str/join [low-string high-string
;;                                         dist-n " = Uniform(low=" low ", high=" high ")\n"])]
;;              (vector dist-n dist-string))

;;           "exponential"
;;            (let [[l l-string] (tf-primitive (first (rest expr)))
;;                  dist-n (gensym "x")
;;                  dist-string (str/join [l-string
;;                                         dist-n " = Exponential(rate=" l ")\n"])]
;;              (vector dist-n dist-string))

          ;;; discrete
          "discrete"  ;; translate to categorical in tf
           (let [[p p-string] (tf-primitive (first (rest expr)))
                 dist-n (gensym "x")
                 dist-string (str/join [p-string
                                        dist-n " = Categorical(p=" p ")\n"])]
             (vector dist-n dist-string))

;;           "poisson"
;;            (let [[l l-string] (tf-primitive (first (rest expr)))
;;                  dist-n (gensym "x")
;;                  dist-string (str/join [l-string
;;                                         dist-n " = Poisson(" l ")\n"])]
;;              (vector dist-n dist-string))
    ))


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
                  pdf-s (str/join [pdf-n " = "  var-e ".log_pdf( " var-n ") # from observe  \n"])]
                 (update-v-p var-n pdf-n)  ;; update the atomn
                 (recur (rest var-list)
                        (str/join [var-string pdf-s])))

            ;; sample vertices
            (let [var-string (str/join [str-prog var-s
                                         var-n " = " var-e ".sample()   #sample \n"])
                  pdf-n (gensym "p")
                  pdf-s (str/join [pdf-n " = " var-e ".log_pdf( " var-n ") # from prior\n"])]
                (update-v-p var-n pdf-n)  ;;update the atom
                (recur (rest var-list)
                        (str/join [var-string pdf-s])))))))))


;;; add all the log-pdf of likelihood together
(defn tf-joint-log-pdf [foppl-query]
   (let [add-n (gensym "p")
         pdf-n (vals @vertice-p)
         add-s (str/join [add-n " = " (python-add pdf-n) " # total log joint \n "])]
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
  (str/join [ "import torch \n"
              "import numpy as np  \n"
              "from torch.autograd import Variable  \n"
              "from Distributions import *  \n"
              "import " expr " as " e "\n"]))

;;; deal with the E
;; output [s1, s2] two string, one declare, one init and run
(defn eval-E [foppl-query]
  (let [E (second foppl-query)
        E-expr (:body E)
        G (first foppl-query)]
;;     (prn E-expr)
    (cond (not (atom? E-expr))   ;; returning sequence
          (let [[s1-n s1] (tf-primitive E-expr)
                s2 (str/join ["print(" s1-n ")\n"])]
            ;(prn s1-n s1)
            (vector s1 s2))

          (contains? (first G) E-expr)  ;; random varibales
          (let [s1 ""
                s2 (str/join["print(" E-expr ")\n"])]
            (vector s1 s2))

          (atom? E-expr)  ;;number, string
          (let [s1 ""
                s2 (str/join["print(" E-expr ")\n"])]
            (vector s1 s2))

          :else "Not Match!")))


;;; __main
(defn compile-query [foppl-query]
  (reset! vertice-p {})
  (let [heading (add-heading "HMC" "HMC \n")
        declare-s (tf-var-declare foppl-query)
        [pdf-n pdf-s] (tf-joint-log-pdf foppl-query)
        f-head (str/join ["def f(x): \n"])   ; need to input actual x
        f-return (str/join ["return " pdf-n "\n"])   ;return grad as well
        [declare-E run-E] (eval-E foppl-query)]
    (str/join [heading
               f-head
               declare-s
               pdf-s
               "# call python process function \n"
               f-return
;;                "# printing original E in foppl: \n"
;;                declare-E
;;                run-E
               ])))




;;;; test case, put here easy to test

;; 1d gaussian
(def one-gaussian
  (foppl-query
    (let [x (sample (normal 1.0 5.0))]
      (observe (normal x 2.0) 7.0)
      x)))
(print-graph (first one-gaussian))
(spit "./output-pytorch/one-gauss.py" (compile-query one-gaussian))

;; ;; bivariate gaussian
;; (def bi-gauss
;;   (foppl-query
;;     (let [mu [0.0 0.0]
;;           cov [[1.0 0.8]
;;                [0.8 1.0]]
;;           x (sample (mvn mu cov))
;;           y [7.0 7.0]]
;;       (observe (mvn x cov) y)
;;       x)))
;; (print-graph (first bi-gauss))
;; (spit "./output-pytorch/bi-gauss.py" (compile-query bi-gauss))

;; ;;; linear regression
;; (def lr-src
;;   (foppl-query
;;     (defn observe-data [_ data slope bias]
;;                         ;;_ loop index
;;       					        ;;data value
;;       					        ;;slop and bias are the real args
;;       (let [xn (first data)
;;             yn (second data)
;;             zn (+ (* slope xn) bias)]
;;         (observe (normal zn 1.0) yn)
;;         (rest (rest data))))

;;     (let [slope (sample (normal 0.0 10.0))
;;           bias  (sample (normal 0.0 10.0))
;;           data (vector
;;                  1.0 2.1 2.0 3.9 3.0 5.3)]
;;                  ;4.0 7.7 5.0 10.2 6.0 12.9)]
;;       (loop 3 data observe-data slope bias)
;;        (vector slope bias))))
;; (print-graph (first lr-src))
;; (spit "./output-pytorch/lr-src.py" (compile-query lr-src))

;; (def if-src
;;   (foppl-query
;;     (let [x (sample (normal 0 1))]
;;       (if (> x 0)
;;         (observe (normal 1 1) 1)
;;         (observe (normal -1 1) 1))
;;       x)))
;; (print-graph (first if-src))

;; (def src-nop
;;   (foppl-query
;;     (let [x (sample (flip 0.5))]
;;       (if x
;;         1
;;         (sample (normal 0 1))))))
;; (print-graph (first src-nop))

;; (def src-nop2
;;   (foppl-query
;;     (let [x (sample (flip 0.5))
;;           smp (sample (normal 0 1))]
;;       (if x
;;         1
;;         (observe (normal smp 1) 7))
;;       )))
;; (print-graph (first src-nop2))


;; (def src2
;;   (foppl-query
;;     (let [x (sample (discrete [0.2 0.3 0.5]))]
;;       (observe (normal x 2) 5)
;;       x)))
;; (print-graph (first src2))


;; (def gmm
;;   (foppl-query
;;     (let [mu (vector -5 5)
;;           obs (vector -7 7)
;;           z (sample (discrete [0.3 0.7]))]
;;       (observe (normal (get mu z) 2) (get obs z))
;;       (vector z (get mu z)))))

;; (def gmm-src
;;   (foppl-query

;;     (defn obs-fn [_ data mu pi]
;;       (let [k (sample (discrete pi))]
;;         (observe (normal (nth mu k) 2) (first data))
;;         (rest data)))

;;     (let [mu (vector (sample (normal 0 2))
;;                      (sample (normal 0 2)))
;;           pi [0.5 0.5]
;;           data [-2.0  -2.5  -1.7  -1.9  -2.2  -2.8  -3  -2.1 -1.2  -1.5  2.0  1.9  2.5  1.5  2.2  3  1.2  2.8  2.3  2.7]
;;           N (count data)]
;;       (loop 5 data obs-fn mu pi)
;;       mu)))
;; (print-graph (first gmm-src))
;; (spit "./output-pytorch/gmm-src.py" (compile-query gmm-src))
