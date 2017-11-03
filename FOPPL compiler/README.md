# compiler to tf

BP output (G, E), where
  G (V, A, P, O) which is (#{}, #{}, {}, {})
  E returning expression

# code structure
tf-primitive[expr]
=> output: [var-n var-s], gensym var name and corresponding string.
   also recursive evaluation/translation of sub expression
   current support: logical and math operation, distribution declare, if, nth


tf-var-expr [foppl-query]
=> [[v [var-n var-s]], [...], ...], link each vertice and their corresponding translation of string (build a dictionary)
  v are vertices from V in G;
  [var-n var-s] the product from tf-primitive

tf-var-declare [foppl-query]
=> a huge string of the declaration part of tf computation graph

tf-joint-log-pdf [foppl-query]
=> compute the total joint log pdf of the program
=> [p-n, p-s], name of the node of computing log-pdf and corresponding string


init-order [foppl-query]
=> order of string for tf sess initiating

##### finally
compile-query[foppl-query]
	add-heading
	tf-var-declare
	tf-joint-log-pdf
	declare-E
	sess-declare
	-
	init
	run pdf
	print E
	sess-close
