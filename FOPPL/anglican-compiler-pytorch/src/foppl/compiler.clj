;; change the part of compute log_pdf(for continuous)/log_pmf(for discrete)
;; some problem in nth, python and tf index

(ns foppl.compiler
  (:require ;[foppl.desugar :refer :all]
            [foppl.core :as foppl :refer [foppl-query print-graph]] :reload
            [clojure.string :as str]
            [clojure.pprint :as ppt])
  (:use [anglican runtime emit]
        [clojure.inspector :include (atom?)]))

(def contdist-list ["normal" "mvn" "beta" "cauchy" "dirichlet" "exponential"
                    "gamma" "half_cauchy" "log_normal" "uniform"])
(def discdist-list ["discrete" "categorical" "poisson" "bernoulli"
                    "multinomial"])


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

;(to-python-list (vector 1 2 3))

(defn python-add [arg]
;;   (let [s "tf.add_n(["
;;         rest-arg (str/join "," arg)]
;;     (str/join [s, rest-arg, "])\n"])))
  (let [add-str (str/join " + " arg)]
    (str/join [" " add-str " "])))
;(python-add (vector 1 2 3))


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
                               r-n " = " elem-n "[int(" num-n ")]\n"])]
            (vector r-n r-s))
    ))


(defn convert-dist [expr]
  (case (name (first expr))
        ;; continuous
          "normal"
           (let [[mu mu-string] (tf-primitive (first (rest expr)))
                 [sigma sigma-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [mu-string sigma-string
                                        dist-n " = dist.Normal(mu=" mu ", sigma="sigma ")\n"])]
             (vector dist-n dist-string))

          "mvn"
           (let [[mu mu-string] (tf-primitive (first (rest expr)))
                 [cov cov-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [mu-string cov-string
                                        dist-n " = dist.MultivariateNormal(mu=" mu ", cov=" cov ")\n"])]
             (vector dist-n dist-string))
          "gamma"
           (let [[alpha alpha-string] (tf-primitive (first (rest expr)))
                 [beta beta-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [alpha-string beta-string
                                        dist-n " = dist.Gamma(alpha=" alpha ", beta=" beta ")\n"])]
             (vector dist-n dist-string))
          "beta"
           (let [[alpha alpha-string] (tf-primitive (first (rest expr)))
                 [beta beta-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [alpha-string beta-string
                                        dist-n " = dist.Beta(alpha=" alpha ", beta=" beta ")\n"])]
             (vector dist-n dist-string))
          "cauchy"
           (let [[mu mu-string] (tf-primitive (first (rest expr)))
                 [gamma gamma-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [mu-string gamma-string
                                        dist-n " = dist.Cauchy(mu=" mu ", gamma=" gamma ")\n"])]
             (vector dist-n dist-string))
          "dirichlet"
           (let [[alpha alpha-string] (tf-primitive (first (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [alpha-string
                                        dist-n " = dist.Dirichlet(alpha=" alpha ")\n"])]
             (vector dist-n dist-string))
          "exponential"
           (let [[lambda lambda-string] (tf-primitive (first (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [lambda-string
                                        dist-n " = dist.Exponential(lambda=" lambda")\n"])]
             (vector dist-n dist-string))
          "log_normal"
           (let [[mu mu-string] (tf-primitive (first (rest expr)))
                 [sigma std-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [mu-string std-string
                                        dist-n " = dist.LogNormal(mu=" mu ", sigma=" std ")\n"])]
             (vector dist-n dist-string))
          "uniform"
           (let [[a a-string] (tf-primitive (first (rest expr)))
                 [b b-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [a-string b-string
                                        dist-n " = dist.Uniform(a=" a ", b=" b ")\n"])]
             (vector dist-n dist-string))
          "half_cauchy"
           (let [[mu mu-string] (tf-primitive (first (rest expr)))
                 [gamma gamma-string] (tf-primitive (second (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [mu-string gamma-string
                                        dist-n " = dist.HalfCauchy(mu=" mu ", gamma=" gamma ")\n"])]
             (vector dist-n dist-string))

          ;; discrete
          "categorical"  ;; translate to categorical in tf
           (let [[p p-string] (tf-primitive (first (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [p-string
                                        dist-n " = dist.Categorical(ps=" p ")\n"])]
             (vector dist-n dist-string))
          "bernoulli"
           (let [[ps p-string] (tf-primitive (first (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [p-string
                                        dist-n " = dist.Bernoulli(ps=" ps ")\n"])]
             (vector dist-n dist-string))
          "poisson"
           (let [[lam lam-string] (tf-primitive (first (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [lam-string
                                        dist-n " = dist.Poisson(lam="lam ")\n"])]
             (vector dist-n dist-string))
          "multinomial"   ;; Distribution over counts for `n` independent `Categorical(ps)` trials.
           (let [[ps ps-string] (tf-primitive (first (rest expr)))
                 [n  n-string ] (tf-primitive (second (rest expr)))
                 dist-n (gensym "dist")
                 dist-string (str/join [ps-string n-string
                                        dist-n " = dist.Multnomial(ps="ps ", n " = n ")\n"])]
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
          ("normal" "mvn" "beta" "cauchy" "dirichlet" "exponential" "gamma" "half_cauchy" "log_normal" "uniform"
           "discrete" "categorical" "poisson" "bernoulli" "multinomial")
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
                    pdf-s (str/join [pdf-n " = " dist-n ".logpdf(" var-n ") if " cond-n " else " 0 " # from observe with if  \n"])]
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
                    pdf-s (str/join [pdf-n " = "  var-e ".logpdf(" var-n ") # from observe  \n"])]
                  (update-v-p var-n pdf-n)  ;; update the atomn
                  (recur (rest var-list)
                         (str/join [var-string pdf-s])
                         (str/join [samples-string]))))

            ;; sample vertices
            (let [[var-e var-s]  (tf-primitive expr)   ;final return would always be the dist obj
                  var-string (str/join [str-prog var-s
                                        ; var-n " = " var-e ".sample()   #sample \n"])
                                         var-n " =  state['" var-n "']   # get the x from input arg\n"])

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
;(defn add-heading-old [expr e]
;  (str/join [ "import torch \n"
;              "import numpy as np  \n"
;              "from torch.autograd import Variable  \n"
;              "from Distributions import *  \n"
;              "import " expr " as " e "\n"]))

(defn add-heading []
  (str/join [ "import torch \n"
              "import numpy as np  \n"
              "from torch.autograd import Variable  \n"
              "import pyfo.distributions as dist\n"
              "import pyfo.inference as interface\n"]))

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

          :else "No Match!")))

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
  (into [] (concat (get-discdist-vars foppl-query) (get-piecewise-vars foppl-query))))

(defn get-cont-vars [foppl-query]
  (let [all-vars (get-vars foppl-query)
        disc-vars (get-disc-vars  foppl-query)]
    (remove #(contains? disc-vars %) all-vars)))

(defn get-ordered-vars [foppl-query]
  (concat (get-cont-vars foppl-query) (get-disc-vars foppl-query)))

(defn create-method
  ([name body]
    (let [body (if (vector? body) (str/join "\n" body) body)
          body (str/replace body #"\n\n" "\n")]
      (str/join ["\t@classmethod\n\tdef " name "(self):\n\t\t"
        (str/replace body #"\n" "\n\t\t") "\n\n"])))
  ([name arg body]
    (let [arg (if (str/blank? arg) "self" (str/join ["self, " arg]))
          body (if (vector? body) (str/join "\n" body) body)
          body (str/replace body #"\n\n" "\n")]
      (str/join ["\t@classmethod\n\tdef " name "(" arg "):\n\t\t"
        (str/replace body #"\n" "\n\t\t") "\n\n"]))))

(defn make-list-of-strings [items]
  (if (empty? items)
    "[]"
    (str/join ["['" (str/join "', '" items) "']"])))

(defn make-return-list [items]
  (str/join ["return " (make-list-of-strings items)]))

;;; output into strings
(defn gen-vars [foppl-query]
  "output variable array of RVs"
  (create-method "gen_vars" (make-return-list (get-vars foppl-query))))
;  (str/join ["\tdef gen_vars(self):\n"
;;              "# generate all unobserved variables \n"
;             "\t\treturn ['" (str/join "', '" (get-vars foppl-query)) "'] # list\n\n"
;             ]))

(defn gen-ordered-vars [foppl-query]
  "output ordered variable array of RVs"
  (create-method "gen_ordered_vars"
    (make-return-list (get-ordered-vars foppl-query))))
;  (str/join ["\tdef gen_ordered_vars(self):\n"
;;              "# generate all unobserved variables \n"
;             "\t\treturn ['" (str/join "', '" (get-ordered-vars foppl-query)) "'] # need to modify output format\n\n"
;             ]))

(defn gen-disc-vars [foppl-query]
  "output discrete and piecewise variable array of RVs"
  (create-method "gen_disc_vars"
    (make-return-list (get-disc-vars foppl-query))))
;  (str/join ["\tdef gen_disc_vars(self):\n"
;;              "# generate discrete and piecewise variables \n"
;             "\t\treturn ['" (str/join "', '" (get-disc-vars foppl-query)) "'] # need to modify output format\n\n"
;             ]))

(defn gen-cont-vars [foppl-query]
  "output continuous variable array of RVs"
  (create-method "gen_cont_vars"
    (make-return-list (get-cont-vars foppl-query))))
;  (str/join ["\tdef gen_cont_vars(self):\n"
;;              "# generate continuous variables \n"
;             "\t\treturn ['" (str/join "', '" (get-cont-vars foppl-query)) "'] # need to modify output format\n\n"
;             ]))

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

(defn gen-gensym-var [foppl-query]
  (str/join [""
;;              "# output var map {gensym var-name: index in x} \n"
             "var_x_map = "  (get-gensym-var foppl-query) " # need to modify output format\n\n"
             ]))


(defn gen-samples-pdf [foppl-query]
  "output functions to gen samples; compute pdf and grad "
  (let [[declare-s declare-prior-samples] (tf-var-declare foppl-query)
        gen-samples-s (create-method "gen_prior_samples"
                                    [declare-prior-samples
                                     "state = {}"
                                     "for _gv in self.gen_vars():"
                                     "\tstate[_gv] = locals()[_gv]"
                                     "return state # dictionary"])
;       gen-samples-s (str/replace
;                                   (str/join ["\tdef gen_prior_samples(self):\n"
;                                    declare-prior-samples
;                                    "state = self.gen_vars() \n"
;                                    "state = locals()[state[0]]\n"
;                                    "return state # list \n\n"]) #"\n" "\n\t\t")
        [pdf-n pdf-s] (tf-joint-log-pdf foppl-query)
        var-cont (get-cont-vars foppl-query)
        grad-s (str/join ["if compute_grad:\n"
                          "\tgrad = torch.autograd.grad(" pdf-n ", var_cont)[0] # need to modify format"
                          ])
        gen-pdf-s (create-method "gen_pdf" "state"
                         [declare-s
                          pdf-s
                          (str/join ["return " pdf-n " # need to modify output format"])])
;        gen-pdf-s (str/replace
;                      (str/join ["\tdef gen_pdf(self, state):\n"
;                                 declare-s
;                                 pdf-s
;                                 "return " pdf-n " # need to modify output format\n\n"])
;                                 #"\n" "\n\t\t")
                                 ]
    [gen-pdf-s gen-samples-s]))

(defn pretty-format-graph [G]
  (let [[V A P O] G]
    (str/join ["Vertices V:\n" V
    "\nArcs A:\n" A
    "\nConditional densities P:\n"
    (str/join "\n" (for [[k v] P] (str k " -> " v)))
    "\nObserved values O:\n"
    (str/join "\n" (for [[k v] O] (str k " -> " v)))])))

(defn generate-doc-str [foppl-query]
  (let [s (pretty-format-graph (first foppl-query))
        s (str/replace s #"\n" "\n\t")]
    (str/join ["\t'''\n\t" s "\n\t'''\n\n"])))

;;; __main
(defn compile-query [foppl-query]
  (reset! vertice-p {})
  (let [heading (add-heading)
        [gen-pdf-s gen-samples-s] (gen-samples-pdf foppl-query)
        ]
        ;[declare-E run-E] (eval-E foppl-query)]
    (str/join [heading
               "\nclass model(interface):\n"
               (generate-doc-str foppl-query)
               (gen-vars foppl-query)
               (gen-cont-vars foppl-query)
               (gen-disc-vars foppl-query)
               "\t# prior samples \n"
               gen-samples-s
               "\t# compute pdf \n"
               gen-pdf-s
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
(spit "./output-pytorch/onedgaussmodel.py" (compile-query one-gaussian))


;; (def if-src
;;   (foppl-query
;;     (let [x (sample (normal 0 1))]
;;       (if (> x 0)
;;         (observe (normal 1 1) 1)
;;         (observe (normal -1 1) 1))
;;       x)))
;; (print-graph (first if-src))
;; (spit "./output-pytorch/if-src-model.py" (compile-query if-src))

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
          z (sample (categorical [0.3 0.7]))]
      (observe (normal (get mu z) 2) (get obs z))
      (vector z (get mu z)))))
(print-graph (first gmm))
(spit "./output-pytorch/ggm.py" (compile-query gmm))

(def poi-src
  (foppl-query
    (let [a (poisson 2)
          b (poisson 7)
          d (uniform-discrete (sample a) (sample b))
          e (sample d)]
      e)))
(print-graph (first poi-src))

(def poi-src2
  (foppl-query
    (let [a (sample (poisson 2))
          b (sample (poisson 7))
          d (uniform-discrete a b)
          e (sample d)]
      e)))
(print-graph (first poi-src2))
(spit "./output-pytorch/poi-src.py" (compile-query poi-src2))

