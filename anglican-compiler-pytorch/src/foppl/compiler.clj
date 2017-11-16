;; change the part of compute log_pdf(for continuous)/log_pmf(for discrete)
;; some problem in nth, python and tf index

(ns foppl.compiler
  (:require ;[foppl.desugar :refer :all]
            [foppl.core :as foppl :refer [foppl-query print-graph]] :reload
            [clojure.string :as str]
            [clojure.pprint :as ppt])
  (:use [anglican runtime emit]
        [clojure.inspector :include (atom?)]))

(def contdist-list ["normal" "mvn"])
(def discdist-list ["discrete"])


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


(declare tf-primitive)

(defn convert-primitive-opt [expr]
  (case (name (first expr))
          "not"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                less-n (gensym "x")
                less-s (str/join [fir-expr
                                  less-n " = not logical_trans(" fir-n ")\n"])]
            (vector less-n less-s))

          "and"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = logical_trans( " fir-n " and " sec-n ")\n"])]
            (vector less-n less-s))
          "or"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = logical_trans( " fir-n " or " sec-n ")\n"])]
            (vector less-n less-s))

          ">"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = logical_trans( " fir-n " > " sec-n ")\n"])]
            (vector less-n less-s))
          ">="
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = logical_trans( " fir-n " >= " sec-n ")\n"])]
            (vector less-n less-s))
          "<"
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = logical_trans( " fir-n " < " sec-n ")\n"])]
            (vector less-n less-s))
          "<="
          (let [[fir-n fir-expr] (tf-primitive (nth expr 1))
                [sec-n sec-expr] (tf-primitive (nth expr 2))
                less-n (gensym "x")
                less-s (str/join [fir-expr sec-expr
                                  less-n " = logical_trans( " fir-n " <= " sec-n ")\n"])]
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
                 dist-n (gensym "dist")
                 dist-string (str/join [mu-string std-string
                                        dist-n " = Normal(mean=" mu ", std=" std ")\n"])]
             (vector dist-n dist-string))

          "mvn"
           (let [[mu mu-string] (tf-primitive (first (rest expr)))
                 [cov cov-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [mu-string cov-string
                                        dist-n " = MultivariateNormal(mean=" mu ", cov=" cov ")\n"])]
             (vector dist-n dist-string))


          ;;; discrete
          "discrete"  ;; translate to categorical in tf
           (let [[p p-string] (tf-primitive (first (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [p-string
                                        dist-n " = Categorical(p=" p ")\n"])]
             (vector dist-n dist-string))

    ))


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

          ;;; distribution
          ("normal" "mvn" "discrete")
          (convert-dist expr)

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

;;         (number? expr)
;;         (let [e-n (gensym "c")
;;               e-s (str/join[e-n "= VariableCast(" expr")\n"])]
;;           (vector e-n e-s))

        :else (vector expr "")))


;;; combine tf-var-expr and tf-var-declare
;;; translate all vertices to py source code
(defn tf-var-declare [foppl-query]
  (let [P (nth (first foppl-query) 2)
        O (nth (first foppl-query) 3)]
    (loop [var-list (foppl/topo-sort foppl-query)
           str-prog ""
           prior-samples ""]
      ;(prn var-list)
      (if (empty? var-list)
;;         str-prog
        [str-prog prior-samples]
        (let [var-n (first var-list)
              expr (:body (get P var-n))]
          (if (contains? O var-n)

            ;; observe
            (if (identical? (name (first expr)) "if")
              ;;; observe start with if
              (let [[cond-n cond-s] (tf-primitive (nth expr 1))
                    [dist-n dist-s] (tf-primitive (nth expr 2))
                    o-value (get O var-n)
                    [o-n o-s] (tf-primitive o-value)
                    var-string (str/join [str-prog cond-s dist-s o-s
                                        var-n " = " o-n " \n"])
                    samples-string (str/join [prior-samples cond-s dist-s o-s
                                              var-n " = " o-n " \n"])
                    pdf-n (gensym "p")
                    pdf-s (str/join [pdf-n " = " dist-n ".logpdf( " var-n ") if " cond-n " else " 0 " # from observe with if  \n"])]
                 (update-v-p var-n pdf-n)  ;; update the atomn
                 (recur (rest var-list)
                        (str/join [var-string pdf-s])
                        (str/join [samples-string])))
              ;;; observe not start with if
              (let [[var-e var-s]  (tf-primitive expr) ;final return would always be the dist obj
                    o-value (get O var-n)
                    [o-n o-s] (tf-primitive o-value)
                    var-string (str/join [str-prog var-s o-s
                                        var-n " = " o-n " \n"])
                    samples-string (str/join [prior-samples var-s o-s
                                              var-n " = " o-n " \n"])
                    pdf-n (gensym "p")
                    pdf-s (str/join [pdf-n " = "  var-e ".logpdf( " var-n ") # from observe  \n"])]
                  (update-v-p var-n pdf-n)  ;; update the atomn
                  (recur (rest var-list)
                         (str/join [var-string pdf-s])
                         (str/join [samples-string]))))

            ;; sample vertices
            (let [[var-e var-s]  (tf-primitive expr)   ;final return would always be the dist obj
                  var-string (str/join [str-prog var-s
                                        ; var-n " = " var-e ".sample()   #sample \n"])
                                         var-n " =  Xs[var_x_map.get('" var-n "')]   # get the x from input arg\n"])

                  samples-string (str/join [prior-samples var-s
                                        var-n " = " var-e ".sample()   #sample \n"])

                  pdf-n (gensym "p")
                  pdf-s (str/join [pdf-n " = " var-e ".logpdf( " var-n ") # from prior\n"])]
                (update-v-p var-n pdf-n)  ;;update the atom
                (recur (rest var-list)
                       (str/join [var-string pdf-s])
                       (str/join [samples-string])))))))))

;; (identical? (name (first '(if (< x 0) (normal 1 1)))) "if")
;; (tf-primitive  '(and (< x22697 1) (> x22697 0)))


;;; add all the log-pdf of likelihood together
(defn tf-joint-log-pdf [foppl-query]
   (let [;add-n (gensym "p")
         add-n "logp"
         pdf-n (vals @vertice-p)
         add-s (str/join [add-n " = " (python-add pdf-n) " # total log joint \n"])]
     (vector add-n add-s)))
;; (tf-joint-log-pdf foppl-src00)



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

;;; get all unobserved variables, disc vars, cont vars
(defn get-vars [foppl-query]
  (let [G (first foppl-query)
        V (first G)
        O (nth G 3)]
    (loop [all-var []
           var-list V]
      (if (empty? var-list)
        all-var
        (if (contains? O (first var-list))
          (recur all-var (rest var-list))
          (recur (conj all-var (first var-list)) (rest var-list)))))))

(defn get-discdist-vars [foppl-query]
  (let [all-var (get-vars foppl-query)
        P (nth (first foppl-query) 2)]
    (loop [discdist-vars []
           var-list all-var]
      (if (empty? var-list)
        discdist-vars
        (let [var-n (first var-list)
              dist-type (first (:body (get P var-n)))]
          (if (contains? discdist-list var-n)
            (recur (conj discdist-vars var-n) (rest var-list))
            (recur discdist-vars (rest var-list))))))))

(defn get-piecewise-vars [foppl-query]
  [])

(defn get-disc-vars [foppl-query]
  (conj (get-discdist-vars foppl-query) (get-piecewise-vars foppl-query)))

(defn get-cont-vars [foppl-query]
  (let [all-vars (get-vars foppl-query)
        disc-vars (get-disc-vars  foppl-query)]
    (remove #(contains? disc-vars %) all-vars)))

(defn get-ordered-vars [foppl-query]
  (concat (get-cont-vars foppl-query) (get-disc-vars foppl-query)))

; output strings
(defn gen-ordered-vars [foppl-query]
  "output variable array of RVs"
  (str/join ["def gen_ordered_vars():\n"
;;              "# generate all unobserved variables \n"
             "return [" (str/join "," (get-ordered-vars foppl-query)) "] # need to modify output format\n\n"
             ]))

(defn gen-disc-vars [foppl-query]
  "output discrete and piecewise variable array of RVs"
  (str/join ["def gen_disc_vars():\n"
;;              "# generate discrete and piecewise variables \n"
             "return [" (str/join "," (get-disc-vars foppl-query)) "] # need to modify output format\n\n"
             ]))

(defn gen-cont-vars [foppl-query]
  "output continuous variable array of RVs"
  (str/join ["def gen_cont_vars():\n"
;;              "# generate continuous variables \n"
             "return [" (str/join "," (get-cont-vars foppl-query)) "] # need to modify output format\n\n"
             ]))

;;; prior samples
(defn get-gensym-var [foppl-query]
  "output var map {gensym var-name: index in x}"
  (let [ordered-var (get-ordered-vars foppl-query)]
    (loop [gensym-var-map {}
           i 0
           var-list ordered-var]
      (if (empty? var-list)
        gensym-var-map
        (recur (assoc gensym-var-map (first var-list) i)
               (inc i)
               (rest var-list))))))
(let [tmp (get-gensym-var if-src)]
  tmp)


(defn gen-gensym-var [foppl-query]
  (str/join [""
;;              "# output var map {gensym var-name: index in x} \n"
             "var_x_map = "  (get-gensym-var foppl-query) " # need to modify output format\n\n"
             ]))


(defn gen-samples-pdf [foppl-query]
  "output functions to gen samples; compute pdf and grad "
  (let [[declare-s declare-prior-samples] (tf-var-declare foppl-query)
        gen-samples-s (str/join ["def gen_prior_samples():\n"
                                   declare-prior-samples
                                   "Xs = gen_ordered_vars() \n"
                                   "return Xs # need to modify output format\n\n"])
        [pdf-n pdf-s] (tf-joint-log-pdf foppl-query)
        var-cont (get-cont-vars foppl-query)
        grad-s (str/join ["if compute_grad:\n"
                          "grad = torch.autograd.grad("pdf-n ", var_cont)[0] # need to modify format \n"
                          ])
        gen-pdf-s (str/join ["def gen_pdf(Xs, compute_grad = True):\n"
                                   declare-s
                                   pdf-s
                                   "var_cont = gen_cont_vars() \n"
                                   grad-s
                                   "return " pdf-n", grad # need to modify output format\n\n"])]
    [gen-pdf-s gen-samples-s]))

;; (print (gen-samples-pdf if-src))



;;; __main
(defn compile-query [foppl-query]
  (reset! vertice-p {})
  (let [heading (add-heading "HMC" "HMC \n")
        [gen-pdf-s gen-samples-s] (gen-samples-pdf foppl-query)]
        ;[declare-E run-E] (eval-E foppl-query)]
    (str/join [;heading
               (gen-gensym-var foppl-query)
               "# prior samples \n"
               gen-samples-s
                "# compute pdf \n"
                gen-pdf-s
               (gen-ordered-vars foppl-query)
               (gen-cont-vars foppl-query)
               (gen-disc-vars foppl-query)
               ])))


;;;;;;;;;;;;;;;;;;;;;;;; test case, put here easy to test

;; 1d gaussian
(def one-gaussian
  (foppl-query
    (let [x (sample (normal 1.0 5.0))
          xx (+ x 1)]
      (observe (normal xx 2.0) 7.0)
      xx)))
(print-graph (first one-gaussian))
(:body (second one-gaussian))
(spit "./output-pytorch/one-gauss-model.py" (compile-query one-gaussian))


(def if-src
  (foppl-query
    (let [x (sample (normal 0 1))]
      (if (> x 0)
        (observe (normal 1 1) 1)
        (observe (normal -1 1) 1))
      x)))
(print-graph (first if-src))
(spit "./output-pytorch/if-src-model.py" (compile-query if-src))

;; (def tmp (nth (first if-src) 2))
;; (get tmp 'x22563)

;; (def tmp-map (atom {}))
;; ;; (swap! vertice-p assoc k v)
;; (type tmp-map)


(def if-src1
  (foppl-query
    (let [x1 (sample (normal 0 1))
          x2 (sample (normal 0 1))]
      (if (> x1 0)
        (observe (normal x2 1) 1)
        (observe (normal -1 1) 1))
      x1)))
(print-graph (first if-src1))



(def if-src2
  (foppl-query
    (let [x (sample (normal 0 1))]
      (if (> x 0)
        (if (< x 1)
          (observe (normal 0.5 1) 1)
          (observe (normal 2 1) 1))
        (if (> x -1)
          (observe (normal -0.5 1) 1)
          (observe (normal -2 1) 1)))
      x)))
(print-graph (first if-src2))
(spit "./output-pytorch/if-src2-model.py" (compile-query if-src2))


(def gmm
  (foppl-query
    (let [mu (vector -5 5)
          obs (vector -7 7)
          z (sample (discrete [0.3 0.7]))]
      (observe (normal (get mu z) 2) (get obs z))
      (vector z (get mu z)))))
(print-graph (first gmm))
