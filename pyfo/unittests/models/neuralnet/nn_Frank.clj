(def latent-dim 2)

(def hidden-dim 10)

(def output-dim 5)

<<<<<<< HEAD
;(require '[clojure.core.matrix :as mat :refer [mmul add mul div sub]])

(defn append-gaussian [_ v]
  (conj v (sample (normal 0.0 1.0))))

(defn make-latent-vector [_]
  (loop latent-dim [] append-gaussian))

(defn make-hidden-vector [_]
  (loop hidden-dim [] append-gaussian))

(defn make-output-vector [_]
=======
(defn append-gaussian [_ v]
  (conj v (sample (normal 0.0 1.0))))

(defn make-latent-vector []
  (loop latent-dim [] append-gaussian))

(defn make-hidden-vector []
  (loop hidden-dim [] append-gaussian))

(defn make-output-vector []
>>>>>>> e12de2fdbd86ebef38601a9b7588e34045844247
  (loop output-dim [] append-gaussian))

(defn append-latent-vector [_ M]
  (conj M (make-latent-vector)))

(defn append-hidden-vector [_ M]
  (conj M (make-hidden-vector)))

(defn append-output-vector [_ M]
  (conj M (make-output-vector)))

(defn relu [v]
<<<<<<< HEAD
  (mul (mat/ge v 0.0) v))

(defn sigmoid [v]
  (div 1.0 (add 1.0 (mat/exp (sub 0.0 v)))))

(defn append-flip [i v p]
  (conj v (sample (flip (nth p i)))))
=======
  (matrix/mul (matrix/ge v 0.0) v))

(defn sigmoid [v]
  (matrix/div 1.0 (matrix/add 1.0 (matrix/exp (matrix/sub 0.0 v)))))

(defn append-flip [i v p]
  (conj v (sample (binomial (nth p i)))))
>>>>>>> e12de2fdbd86ebef38601a9b7588e34045844247

(let [z (make-latent-vector)

      ;; first: hidden layer
      W (loop hidden-dim [] append-latent-vector)
      b (make-hidden-vector)
<<<<<<< HEAD
      h (relu (add (mmul W z) b))
=======
      h (relu (matrix/add (matrix/mmul W z) b))
>>>>>>> e12de2fdbd86ebef38601a9b7588e34045844247

      ;; output layer
      V (loop output-dim [] append-hidden-vector)
      c (make-output-vector)]
<<<<<<< HEAD
  (loop output-dim [] append-flip (sigmoid (add (mmul V h) c))))
=======
  (loop output-dim [] append-flip (sigmoid (matrix/add (matrix/mmul V h) c))))
>>>>>>> e12de2fdbd86ebef38601a9b7588e34045844247
