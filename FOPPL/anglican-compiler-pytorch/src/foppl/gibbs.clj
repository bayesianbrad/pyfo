(ns foppl.gibbs
  (:require [clojure.set :refer [union intersection]]
            [foppl.core :as foppl]
            [anglican.runtime]
            :reload))

(defn mh-propose [foppl-model trace]
  (let [[[V A P O] E] foppl-model
        rdb (:rdb trace)
        address (rand-nth (keys rdb))
        dist (foppl/get-dist-at-addr trace P address)
        value (anglican.runtime/sample* dist)
        proposal (assoc-in trace [:rdb address] value)]
    (assoc proposal :logprob (foppl/compute-logprob foppl-model proposal))))

(defn mh-step [foppl-model trace]
  (let [proposal (mh-propose foppl-model trace)
        lp (:logprob trace)
        lp2 (:logprob proposal)]
    (if (> (- lp2 lp) (Math/log (rand))) proposal trace)))

(defn gibbs-seq [foppl-model]
  (letfn [(sample-seq [foppl-model trace]
            (lazy-seq
             (let [next-trace (mh-step foppl-model trace)]
               (cons (foppl/get-output foppl-model next-trace)
                     (sample-seq foppl-model next-trace)))))]
    (sample-seq foppl-model (foppl/draw-from-prior foppl-model))))
