(ns foppl.test
  (:require ;[foppl.desugar :refer :all]
            ;[foppl.core :as foppl :refer [foppl-query print-graph]]
            [foppl.core :refer :all]
            [foppl.compiler :refer :all]
            [clojure.pprint :as pprint]
            :reload)
  (:use [anglican runtime]))


;;; first simple model
(def src0
  (foppl-query
    (let [x (sample (normal 1.0 1.0))]
      (observe (normal (+ x 1.0) 1.0) 7.0)
      x)))
(print-graph (first src0))
(spit "./output/src0.py" (compile-query src0))

(print (tf-var-expr src0))
(print (second src0))


;;; only if, without loop

;; sample in if
(def if-src0
  (foppl-query
    (let [x (sample (normal 0.0 5.0))
          new-x (if (<= x 0.0)
                    (vector x (sample (normal x 1.0)))
                    (vector x (sample (normal x 2.0))))]
      (observe (normal (second new-x) 5.0) 7.0)
      new-x)))

(print-graph (first if-src0))
(print "return value: " (:body (second if-src0)))
(spit "./output/if-src0.py" (compile-query if-src0))


;; other version
(def if-src01-a
  (foppl-query
    (let [x (sample (normal 0.0 5.0))]
      (if (< x 0)
        (sample (normal -1 1))
        (sample (normal 1 1))))))

(print-graph (first if-src01-a))

(def if-src01-b
  (foppl-query
    (let [x (sample (normal 0.0 5.0))
          x1 (sample (normal -1 1))
          x2 (sample (normal 1 1))]
      (if (< x 0)
        x1
        x2))))

(print-graph (first if-src01-b))

;; sampel + obs
(def if-src02-a
  (foppl-query
    (let [x (sample (normal 0.0 5.0))]
      (if (< x 0)
        (let [x1 (sample (normal -1 1))]
          (observe (normal x1 1) 1))
        (let [x2 (sample (normal 1 1))]
          (observe (normal x2 1) 1))))))
(print-graph (first if-src02-a))

(def if-src02-b
  (foppl-query
    (let [x (sample (normal 0.0 5.0))
          x1 (sample (normal -1 1))
          x2 (sample (normal 1 1))]
      (if (< x 0)
          (observe (normal x1 1) 1)
          (observe (normal x2 1) 1)))))
(print-graph (first if-src02-b))

;; obs in if => some problem!!!
(def if-src1
  (foppl-query
    (let [x (sample (normal 0.0 1.0))]
      (if (<= x 0.0)
          (observe (normal x 3.0) -5.0)
          (observe (normal x 5.0) 5.))
      x)))

(print-graph (first if-src1))
(spit "./output/if-src1.py" (compile-query if-src1))

(tf-joint-log-pdf if-src1)

(tf-var-expr if-src1)



;;; loop + if

;; only sample in if
(def loop-if-sample1
  (foppl-query
    (defn loop-fn [n x]
      (if (< (last x) 0.0)
        (let [new-x (sample (normal (last x) 1.0))]
          (conj x new-x))
        (let [new-x (sample (normal (last x) 5.0))]
          (conj x new-x))
        ))

    (let [x (sample (normal 0.0 1.0))]
      (loop 2 (vector x) loop-fn))))
(print-graph (first loop-if-sample1))
(print "return value: " (:body (second loop-if-sample1)))
(spit "./output/loop-if-sample1.py" (compile-query loop-if-sample1))

(def loop-if-sample2
  (foppl-query
    (defn loop-fn [n x]
      (let [cur (< x 0.0)
            std (if cur 3.0 5.0)]
        (sample (normal x std))))

    (let [x (sample (normal 0.0 1.0))]
      (loop 3 x loop-fn))))
(print-graph (first loop-if-sample2))
(spit "./output/loop-if-sample2.py" (compile-query loop-if-sample2))


(def loop-if-obs0
  (foppl-query
    (defn loop-fn [n x]
      (let [cur-x (last x)
            cur (< cur-x 0.0)
            std (if cur 3.0 5.0)
            new-x (sample (normal cur-x std))]
        (observe (normal cur-x 5.0) 5.0)
        (conj x new-x)))

    (let [x (sample (normal 0.0 1.0))]
      (loop 3 (vector x) loop-fn))))

(print-graph (first loop-if-obs0))
(spit "./output/loop-if-obs0.py" (compile-query loop-if-obs0))

(def loop-if-obs1
  (foppl-query
    (defn loop-fn [n x]
      (if (< x 0.0)
        (let [new-x (sample (normal -2.0 3.0))]
          (observe (normal x 1.0) 5.0)
          new-x)
        (let [new-x (sample (normal 2.0 5.0))]
          (observe (normal x 2.0) 5.0)
          new-x)))

    (let [x (sample (normal 0.0 1.0))]
      (loop 3 x loop-fn))))


(def loop-if-obs2
  (foppl-query
    (defn loop-fn [n x]
      (let [which (< x 0.0)
            mean (if which -2.0 2.0)
            sd (if which 3.0 5.0)
            obs-sd (if which 1.0 2.0)
            new-x (sample (normal mean sd))]
        (observe (normal x obs-sd) 5.0)
        new-x))

    (let [x (sample (normal 0.0 1.0))]
      (loop 3 x loop-fn))))
(print-graph (first loop-if-obs2))



;(if (< (if (< (if (< x26667 0.0) x26670 x26674) 0.0) x26678 x26682) 0.0)
;  (normal (if (< (if (< x26667 0.0) x26670 x26674) 0.0) x26678 x26682) 1.0))

(print-graph (first loop-if-obs1))
;; (topo-sort loop-if-obs1)
(spit "./output/loop-if-obs1.py" (compile-query loop-if-obs1))




;; (def if-src1
;;   (foppl-query
;;     (let [x (sample (normal 1.0 1.0))]
;;       (if (<= x 0.3)
;;           (observe (normal x 1.0) -5.0)
;;           (observe (normal x 1.0) 5.))
;;       (if (and (>= x 0.0) (<= x 1.0))
;;           (observe (beta x 0.5) 0.8)
;;           (observe (normal x 1.0) 1.))
;;       (if (>= x 0.7)
;;           (observe (gamma x 1.0) 5.0)
;;           (observe (normal x 1.0) 1.))
;;       x)))

;; (print-graph (first if-src1))
;; (spit "./output/if-src1.py" (compile-query if-src1))

