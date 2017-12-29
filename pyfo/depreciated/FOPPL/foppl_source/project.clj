(defproject anglican-lite "0.1.0-SNAPSHOT"
  :description "Compile a stripped-down version of Anglican into a graph"
  :url "https://bitbucket.org/brx/anglican-lite-clj/"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :plugins [[lein-gorilla "0.3.6"]]
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [anglican "1.0.0"]
                 [org.clojure/data.priority-map "0.0.7"]
                 [org.clojure/data.csv "0.1.3"]
                 [net.mikera/core.matrix "0.52.2"]
                 [net.mikera/vectorz-clj "0.44.1"]]
  :target-path "target/%s"
  :jvm-opts ["-Xmx6g" "-Xms4g"]
  :profiles {:uberjar {:aot :all}})
