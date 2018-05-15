;(let [x (sample (uniform 0 1))
;      q 0.5
;      y 0.25]
;  (if (< (- q x) 0)
;    (observe (normal 1 1) y)
;    (observe (normal 0 1) y))
;  (< (- q x) 0))
(let [x (sample (normal 0 1))
      y 1]
  (if ((> x 0))
    (observe (normal (+ x 1) 1) y)
    (observe (normal (- x 1) 1) y))
  )

;Equivilent stan model
;q ~ normal(0, 1);
;     z ~ normal(0,1);
;
;     if (z < q)
;        {y ~ normal(0, 1);
;        target += normal_lpdf(y | 0,1) +  normal_lpdf(q | 0, 1) + normal_lpdf(z | 0,1);}
;     else
;         {y ~ normal(0,2);
;         target += normal_lpdf(y | 0,2) +  normal_lpdf(q | 0, 1) + normal_lpdf(z | 0,1);}
;}