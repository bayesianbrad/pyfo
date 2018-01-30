;;; Date 2018-01-26

(def N 10)

(defn sample-components [_ zs pi]
  (let [z (sample (categorical pi))]
    (conj zs z)))

(defn get-ordered-mu [mu1 mu2]
  (if (< mu1 mu2)
    (vector mu1 mu2)
    (vector mu2 mu1)))

(defn observe-data [n _ ys zs mus]
  (let [y (get ys n)
        z (get zs n)
        mu (get mus z)]
    (observe (normal mu 1) y)
    nil))

(let [ys      (vector -2.0  -2.5  -1.7  -1.9  -2.2
                      1.5  2.2  3  1.2  2.8)
      pi [0.5 0.5]
      zs  (loop N (vector) sample-components pi)
      mu1 (sample (normal 0 100))   ; std = 10
      mu2 (sample (normal 0 100))
      mus (get-ordered-mu mu1 mu2)]
  (loop N nil observe-data ys zs mus)
  (vector mus zs))
