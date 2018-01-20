;(let [mu (vector -5 5)
;      obs (vector -7 7)
;      z (sample (categorical [0.3 0.7]))]
;  (observe (normal (get mu z) 2) (get obs z))
;  (vector z (get mu z)))

;(let [mu (vector -5 5)
;      obs 7
;      z (sample (categorical [0.5 0.5]))]
;  (observe (normal (get mu z) 2) obs)
;  (vector z mu))


;(let [mu1 (sample (normal -5 1))
;      mu2 (sample (normal 5 1))
;      mu (vector mu1 mu2)
;      z (sample (categorical [0.5 0.5]))
;      y 7]
;  (observe (normal (get mu z) 2) y)
;  (vector z mu ))


;; unknown prior mean for the center of each cluster

(defn sample-components [_ zs pi]
  (let [z (sample (categorical pi))]
    (conj zs z)))

(defn observe-data [n _ ys zs mus]
  (let [y (get ys n)
        z (get zs n)
        mu (get mus z)]
    (observe (normal mu 2) y)
    nil))

(let [ys      (vector -2.0  -2.5  -1.7  -1.9  -2.2
                      1.5  2.2  3  1.2  2.8)
      pi [0.5 0.5]
      zs  (loop 10 (vector) sample-components pi)
      mus (vector (sample (normal 0 2))
                  (sample (normal 0 2)))]
  (loop 10 nil observe-data ys zs mus)
  (vector mus zs))