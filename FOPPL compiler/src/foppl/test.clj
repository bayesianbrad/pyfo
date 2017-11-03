(ns foppl.test
  (:require ;[gorilla-plot.core :as plot]
            [foppl.desugar :refer :all]
            [foppl.core :as foppl :refer [foppl-query print-graph]]
            [foppl.compiler :refer :all]
            :reload)
  (:use [anglican runtime]))


;;; first simple model
(def src0
  (foppl-query
    (let [x (sample (normal 1.0 1.0))]
      (observe (normal x 1.0) 7.0)
      x)))
(print-graph (first src0))
(spit "./output/src0.py" (compile-query src0))

;;; linear regression
(def lr-src
  (foppl-query
    (defn observe-data [_ data slope bias]
                        ;;_ loop index
      					;;data value
      					;;slop and bias are the real args
      (let [xn (first data)
            yn (second data)
            zn (+ (* slope xn) bias)]
        (observe (normal zn 1.0) yn)
        (rest (rest data))))

    (let [slope (sample (normal 0.0 10.0))
          bias  (sample (normal 0.0 10.0))
          data (vector
                 1.0 2.1 2.0 3.9 3.0 5.3)]
                 ;4.0 7.7 5.0 10.2 6.0 12.9)]
      (loop 3 data observe-data slope bias)
       (vector slope bias))))
(print-graph (first lr-src))
(spit "./output/lr-src.py" (compile-query lr-src))


(def if-src
  (foppl-query
    (let [x (sample (normal 1.0 1.0))]
      (if (<= x 0.3)
          (observe (normal x 1.0) -5.0)
          (observe (normal x 1.0) 5.))
      (if (and (>= x 0.0) (<= x 1.0))
          (observe (beta x 0.5) 0.8)
          (observe (normal x 1.0) 1.))
      (if (>= x 0.7)
          (observe (gamma x 1.0) 5.0)
          (observe (normal x 1.0) 1.))
      x)))

(print-graph (first if-src))
(spit "./output/if-src.py" (compile-query if-src))


(def hmm-src
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
                   (vector 0.7 0.15 0.15 )) k))

    (defn get-obs-dist [k]
      (nth (vector (normal -1. 1.)
                   (normal  1. 1.)
                   (normal  0. 1.)) k))   ; have some problem in tf when indexing obj

    ;; Function to step through HMM and sample latent state
    (defn hmm-step [n states]
      (let [next-state (sample (discrete (get-trans-params (last states))))]
        (observe (get-obs-dist next-state) (data n))
        (conj states next-state)))

    ;; Loop through the data
    (let [init-state (sample (discrete (get-init-params)))]
      (loop 1 (vector init-state) hmm-step))))

(print-graph (first hmm-src))
(spit "./output/hmm-src.py" (compile-query hmm-src))

;rewirte hmm, change a little bit on indexing
(def hmm-src2
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
                   (vector 0.7 0.15 0.15 )) k))

    (defn get-obs-dist [k]
      (nth (vector (vector -1. 1.)
                   (vector  1. 1.)
                   (vector  0. 1.)) k))

    ;; Function to step through HMM and sample latent state
    (defn hmm-step [n states]
      (let [;cur-state (last states)
             next-state (sample (discrete (get-trans-params (last states))))
            obs-param (get-obs-dist next-state)]
        (observe (normal (nth obs-param 0) (nth obs-param 1)) (data n))  ;;a little bit weird but work!
        (conj states next-state)))

    ;; Loop through the data
    (let [init-state (sample (discrete (get-init-params)))]
      (loop 3 (vector init-state) hmm-step))))

(print-graph (first hmm-src2))
(spit "./output/hmm-src2.py" (compile-query hmm-src2))

;;; hmm in if-else
(def hmm-if-src1
  (foppl-query

    (defn data [n]
      (let [points (vector 0.9 0.8 -0.7 -0.5 -0.025
                           5.0 2.0 0.1 0.0 0.13
                           0.45 6.0 0.2 0.3 -1.0 -1.0)]
        (get points n)))

    (defn hmm-step [n states]
      (let [cur-state (last states)]
        (if (< cur-state 0.)
          (let [next-state (sample (normal 0.0 2.0))]
            (observe (normal next-state 1.0) (data n))
            (conj states next-state))
          (let [next-state (sample (normal 0.0 2.0))]
            (observe (normal next-state 1.0) (data n))
            (conj states next-state)))))

    ;; Main Loop through the data
    (let [init-state (sample (normal 0. 5.))]
      (loop 2 (vector init-state) hmm-step))))

(print-graph (first hmm-if-src1))
(spit "./output/hmm-if-src1.py" (compile-query hmm-if-src1))


(def hmm-if-src2
  (foppl-query

    (defn data [n]
      (let [points (vector 0.9 0.8 -0.7 -0.5 -0.025
                           5.0 2.0 0.1 0.0 0.13
                           0.45 6.0 0.2 0.3 -1.0 -1.0)]
        (get points n)))

    (defn hmm-step [n states]
      (let [cur-state (last states)]
        (if (< cur-state 0.)
          (let [next-state (sample (normal (* 0.1 cur-state) 2.0))]
            (observe (normal next-state 1.0) (data n))
            (conj states next-state))
          (let [next-state (sample (normal (* cur-state cur-state) 2.0))]
            (observe (normal next-state 1.0) (data n))
            (conj states next-state)))))

    ;; Main Loop through the data
    (let [init-state (sample (normal 0. 5.))]
      (loop 2 (vector init-state) hmm-step))))

(spit "./output/hmm-if-src2.py" (compile-query hmm-if-src2))


(def gmm-src

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
              zs (loop 2 (vector) sample-components z-prior)
              likes (loop 3 (vector) sample-likelihoods)]
          (loop 2 nil observe-data ys zs likes)
          zs)))


(print-graph (first gmm-src))
(spit "./output/gmm-src.py" (compile-query gmm-src))
