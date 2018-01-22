(let [mu1 (sample (normal -5 1))
      mu2 (sample (normal 5 1))
      ;sig1 (sample (poisson 2))
      mu (vector mu1 mu2)
      z (sample (categorical [0.1 0.9]))
      y 3]
  (observe (normal (get mu z) 2) y)
  (vector z mu ))
;(defn sample-likelihoods [_ likes]
;      (let [precision (sample (gamma 1.0 1.0))
;            mean (sample (normal 0.0 precision))
;            sigma (/ (sqrt precision))]
;        (conj likes
;              (normal mean sigma))))
;
;    (defn sample-components [_ zs prior]
;      (let [z (sample prior)]
;        (conj zs z)))
;
;    (defn observe-data [n _ ys zs likes]
;      (let [y (nth ys n)
;            z (nth zs n)]
;        (observe (nth likes z) y)
;        nil))
;
;    (let [ys (vector 1.1 2.1 2.0 1.9 0.0 -0.1 -0.05)
;          z-prior (discrete
;                    (sample (dirichlet (vector 1.0 1.0 1.0))))
;          zs (loop 7 (vector) sample-components z-prior)
;          likes (loop 3 (vector) sample-likelihoods)]
;      (loop 7 nil observe-data ys zs likes)
;      zs)