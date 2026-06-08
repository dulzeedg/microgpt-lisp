#lang racket

;; block 
(define names
  (file->lines "names.txt"))

(define uchars
     (list->vector
      (sort
       (remove-duplicates
        (apply append
               (map string->list names))
        char=?)
    char<?))) 

(define BOS (vector-length uchars))

(define vocab-size (+ BOS 1))

(define char->token-table (make-hasheq))
(for ([i (in-range (vector-length uchars))])
  (hash-set! char->token-table (vector-ref uchars i) i))

(define (char->token ch)
  (hash-ref char->token-table ch
            (lambda ()
              (error "char not found"))))

;; block 2
;; Autograd engine
(define (make-value data)
  (vector data ; 0: data
          0.0  ; 1: grad  
          '()  ; 2: children
          '())) ; 3: local-grads

;; Accessors
(define (value-data v) (vector-ref v 0))
(define (value-grad v) (vector-ref v 1))
(define (value-children v) (vector-ref v 2))
(define (value-local-grads v) (vector-ref v 3))

;; Mutators
(define (set-value-data! v x) (vector-set! v 0 x))
(define (set-value-grad! v g) (vector-set! v 1 g))
(define (set-value-children! v xs) (vector-set! v 2 xs))
(define (set-value-local-grads! v xs) (vector-set! v 3 xs))

;; Operations
(define (v+ a b)
  (let ([out (make-value (+ (value-data a) (value-data b)))])
    (vector-set! out 2 (list a b))
    (vector-set! out 3 (list 1.0 1.0))
    out))

(define (v* a b)
  (let ([out (make-value (* (value-data a) (value-data b)))])
    (vector-set! out 2 (list a b))
    (vector-set! out 3 (list (value-data b) (value-data a)))
    out))

(define (vpow x k)
  (let* ([xv (value-data x)]
         [out (make-value (expt xv k))])
    (vector-set! out 2 (list x))
    (vector-set! out 3 (list (* k (expt xv (- k 1)))))
    out))

(define (vlog x)
  (let ([out (make-value (log (value-data x)))])
    (vector-set! out 2 (list x))
    (vector-set! out 3 (list (/ 1.0 (value-data x))))
    out))

(define (v-exp x)
  (let* ([d (exp (value-data x))]
         [out (make-value d)])
    (vector-set! out 2 (list x))
    (vector-set! out 3 (list d)) 
    out))

(define (vneg x)
  (let ([out (make-value (- (value-data x)))])
    (vector-set! out 2 (list x))
    (vector-set! out 3 (list -1.0))
    out))

(define (v- a b)
  (v+ a (vneg b)))

(define (v/ a b)
  (let* ((out (make-value (/ (value-data a) (value-data b)))))
    (vector-set! out 2 (list a b))
    (vector-set! out 3
                 (list
                  (/ 1 (value-data b))             
                  (/ (* -1 (value-data a))
                     (expt (value-data b) 2))))   
    out))

(define (vrelu x)
  (let ([out (make-value (if (> (value-data x) 0)
                             (value-data x)
                             0.0))])
    (set-value-children! out (list x))
    (set-value-local-grads! out (list (if (> (value-data x) 0) 1.0 0.0)))
    out))

;; Topological sort
(define (contains? xs x)
  (cond ((null? xs) #f)
        ((eq? (car xs) x) #t)
        (else (contains? (cdr xs) x))))

(define (build-topo v visited topo)
  (if (hash-has-key? visited v)
      topo
      (begin
        (hash-set! visited v #t)
        (let ((tp topo))
          (for-each
           (lambda (child)
             (set! tp
                   (build-topo child
                               visited
                               tp)))
           (value-children v))
          (cons v tp)))))
;; Back propagation
(define (backward loss)
  (define topo
    (build-topo loss (make-hasheq) '()))

  ;; zero grads
  (for-each (lambda (v)
              (set-value-grad! v 0.0))
            topo)

  ;; seed
  (set-value-grad! loss 1.0)

  ;; propagate
  (for-each
   (lambda (v)
     (define grad (value-grad v))
     (define cs (value-children v))
     (define ls (value-local-grads v))

     (for ([c cs] [l ls])
       (set-value-grad!
        c
        (+ (value-grad c)
           (* grad l)))))
   topo
   ))

;; block 3
;; hyperprams

;; depth of the transformer neural network (number of layers)
(define n-layer 1)

;; width of the network
(define n-embd 16)

;; Block size
(define block-size 16)

;; number of attention heads 4
(define n-head 4)

;; dimension of each head
(define head-dim (quotient n-embd n-head))

;; helper functions
;; normal random using Box-Muller
(define (normal-random mean std)
  (let loop ()
    (let* ((u (- (* 2 (random)) 1))
           (v (- (* 2 (random)) 1))
           (s (+ (* u u) (* v v))))
      (if (or (= s 0) (>= s 1))
          (loop)
          (+ mean (* std (* u (sqrt (/ (* -2 (log s)) s)))))))))

;; parameter creation 
;; matrix of Value objects
;; matrix helpers
(define (param-matrix nout nin #:std [std 0.08])
  (for/vector ([i nout])
    (for/vector ([j nin])
      (make-value (normal-random 0 std)))))
  
(define state-dict (make-hasheq))
(hash-set! state-dict 'wte (param-matrix vocab-size n-embd))
(hash-set! state-dict 'wpe (param-matrix block-size n-embd))
(hash-set! state-dict 'lm-head (param-matrix vocab-size n-embd))

(for ([i (in-range n-layer)])
  (hash-set! state-dict (string->symbol (format "layer~a.attn_wq" i))
             (param-matrix n-embd n-embd))
  (hash-set! state-dict (string->symbol (format "layer~a.attn_wk" i))
             (param-matrix n-embd n-embd))
  (hash-set! state-dict (string->symbol (format "layer~a.attn_wv" i))
             (param-matrix n-embd n-embd))
  (hash-set! state-dict (string->symbol (format "layer~a.attn_wo" i))
             (param-matrix n-embd n-embd))
  (hash-set! state-dict (string->symbol (format "layer~a.mlp_fc1" i))
             (param-matrix (* 4 n-embd) n-embd))
  (hash-set! state-dict (string->symbol (format "layer~a.mlp_fc2" i))
             (param-matrix n-embd (* 4 n-embd))))

(define (params state-dict)
  (apply append
         (for/list ([mat (hash-values state-dict)])
           (apply append
                  (for/list ([row mat])
                    (vector->list row))))))


;; helpers
(define (linear x w)
  (for/vector ([row w])
    (unless (= (vector-length row)
               (vector-length x))
      (error "linear shape mismatch"))

    (let loop ((i 0)
               (acc (make-value 0.0)))
      (if (= i (vector-length row))
          acc
          (loop (+ i 1)
                (v+ acc
                    (v* (vector-ref row i)
                        (vector-ref x i))))))))
(define (v-max vec)
  (let loop ((i 1)
             (m (value-data (vector-ref vec 0))))
    (if (= i (vector-length vec))
        m
        (loop (+ i 1)
              (max m (value-data (vector-ref vec i)))))))

(define (softmax logits)
  (define max-val (v-max logits))
  (define shifted
    (for/vector ([v logits])
      (v+ v (make-value (- max-val)))))
  (define exps
    (for/vector ([v shifted])
      (v-exp v)))
  (define total
    (let loop ((i 0) (acc (make-value 0.0)))
      (if (= i (vector-length exps))
          acc
          (loop (+ i 1)
                (v+ acc (vector-ref exps i))))))
  (for/vector ([e exps])
    (v/ e total)))

(define (rmsnorm x)
  (let loop ((i 0)
             (ms (make-value 0.0)))
    (if (= i (vector-length x))
        (let* ([ms (v/ ms (make-value (vector-length x)))]
               [norm (vpow (v+ ms (make-value 1e-5)) -0.5)])
          (define out (make-vector (vector-length x)))
          (for ([j (in-range (vector-length x))])
            (vector-set! out j
                         (v* (vector-ref x j) norm)))
          out)
        (loop (+ i 1)
              (v+ ms
                  (v* (vector-ref x i)
                      (vector-ref x i)))))))

(define (dot a b)
  (unless (and (vector? a) (vector? b))
    (error "DOT ERROR: inputs must be vectors"))
  (unless (= (vector-length a) (vector-length b))
    (error "DOT ERROR: size mismatch"))

  (let loop ((i 0)
             (acc (make-value 0.0)))
    (if (= i (vector-length a))
        acc
        (loop (+ i 1)
              (v+ acc
                  (v* (vector-ref a i)
                      (vector-ref b i)))))))

(define (slice vec start end)
  (let* ((n (- end start))
         (out (make-vector n)))
    (let loop ((i 0))
      (if (< i n)
          (let ((v (vector-ref vec (+ start i))))
            (unless (and (vector? v) (= (vector-length v) 4))
              (error "BAD SLICE VALUE" v))
            (vector-set! out i v)
            (loop (+ i 1)))
          out))))

;; block 4
;; Attention
(define (attention x li pos-id keys values)

  (define x-norm (rmsnorm x))

  (define Wq (hash-ref state-dict (string->symbol (format "layer~a.attn_wq" li))))
  (define Wk (hash-ref state-dict (string->symbol (format "layer~a.attn_wk" li))))
  (define Wv (hash-ref state-dict (string->symbol (format "layer~a.attn_wv" li))))
  (define Wo (hash-ref state-dict (string->symbol (format "layer~a.attn_wo" li))))

  (define q (linear x-norm Wq))
  (define k (linear x-norm Wk))
  (define v (linear x-norm Wv))

  ;; KV cache update
  (vector-set! (vector-ref keys li) pos-id k)
  (vector-set! (vector-ref values li) pos-id v)

  (define scale (sqrt head-dim))

  (define x-attn (make-vector n-embd))

  (for ([h (in-range n-head)])
    (define hs (* h head-dim))

    (define q-h (slice q hs (+ hs head-dim)))

    (define logits (make-vector (+ pos-id 1)))
    
    ;; compute attention scores
    (for ([t (in-range (+ pos-id 1))])

      (define k-t (vector-ref (vector-ref keys li) t))
      (define k-h (slice k-t hs (+ hs head-dim)))

      (define raw-score
        (dot q-h k-h))

      ;; scale INSIDE graph
      (define score
        (v/ raw-score (make-value scale)))

      (vector-set! logits t score))

    ;; softmax over Value logits
    (define w (softmax logits))

    ;; weighted sum of values
    (for ([j (in-range head-dim)])

      (define acc
        (for/fold ([a (make-value 0.0)])
                  ([t (in-range (+ pos-id 1))])

          (define v-t (vector-ref (vector-ref values li) t))
          (define v-h (slice v-t hs (+ hs head-dim)))

          (v+ a
              (v* (vector-ref w t)
                  (vector-ref v-h j)))))

      (vector-set! x-attn (+ hs j) acc)))

  ;; output projection
  (define x-proj (linear x-attn Wo))

  ;; residual connection (scalar-safe)
  (for ([i (in-range n-embd)])
    (vector-set! x-proj i
                 (v+ (vector-ref x-proj i)
                     (vector-ref x i))))
  x-proj)

;; Multi-layer Perceptron
(define (mlp x li)
  
  ;; layer norm (scalar graph)
  (define x-norm (rmsnorm x))
  ;; fc1 projection
  (define W1
    (hash-ref state-dict
              (string->symbol
               (string-append "layer"
                              (number->string li)
                              ".mlp_fc1"))))

  (define fc1 (linear x-norm W1))

  ;; activation (ReLU)
  (define act (make-vector (* 4 n-embd)))

  (for ([i (in-range (* 4 n-embd))])
    (vector-set! act i
                 (vrelu (vector-ref fc1 i))))

  ;; fc2 projection
  (define W2
    (hash-ref state-dict
              (string->symbol
               (string-append "layer"
                              (number->string li)
                              ".mlp_fc2"))))

  (define fc2 (linear act W2))

  ;; residual connection
  (define out (make-vector n-embd))

  (for ([i (in-range n-embd)])
    (vector-set! out i
                 (v+ (vector-ref fc2 i)
                     (vector-ref x i))))
  out)

;; GPT
(define (gpt token-id pos-id keys values)

  ;; 1. token + position embd
  (define tok-emb
    (vector-ref (hash-ref state-dict 'wte) token-id))

  (define pos-emb
    (vector-ref (hash-ref state-dict 'wpe) pos-id))

  (define x (make-vector n-embd))

  ;; sum embeddings (scalar-safe)
  (for ([i (in-range n-embd)])
    (vector-set! x i
                 (v+ (vector-ref tok-emb i)
                     (vector-ref pos-emb i))))

  ;; initial normalization (still scalar graph)
  (define x0 (rmsnorm x))

  ;; 2. transformer stack
  (let layer-loop ((li 0)
                   (x x0))
    (if (= li n-layer)
        ;; 3. final logits
        (linear x (hash-ref state-dict 'lm-head))
        ;; 4. one transformer layer
        (let* ((x1 (attention x li pos-id keys values))
               (x2 (mlp x1 li)))
          (layer-loop (+ li 1) x2)))))

;; block 5
(define learning-rate 0.01)
(define beta1 0.85)
(define beta2 0.99)
(define eps-adam 1e-8)

(define num-params (length (params state-dict)))
(define param-list-v
  (list->vector (params state-dict)))

;; first moment buffer
(define m
  (make-vector num-params 0.0))

;; second moment buffer
(define v
  (make-vector num-params 0.0))

;; number of training steps
(define num-steps 1000)

(define docs-vec
  (list->vector names))

;; adam optimizer and buffers
;; OPTIMIZER (ADAM)
(define (adam-step params m v step lr beta1 beta2 eps)
  (for ([i (in-range (vector-length params))])
    (define p (vector-ref params i))
    (define g (value-grad p))
    ;; first moment
    (define m-new
      (+ (* beta1 (vector-ref m i))
         (* (- 1 beta1) g)))
    ;; second moment
    (define v-new
      (+ (* beta2 (vector-ref v i))
         (* (- 1 beta2) (* g g))))
    ;; bias correction
    (define m-hat
      (/ m-new (- 1 (expt beta1 (+ step 1)))))

    (define v-hat
      (/ v-new (- 1 (expt beta2 (+ step 1)))))
    ;; update step
    (define update
      (/ m-hat (+ (sqrt v-hat) eps)))
    ;; write back state
    (vector-set! m i m-new)
    (vector-set! v i v-new)

    ;; parameter update (scalar world)
    (set-value-data!
     p
     (- (value-data p)
        (* lr update)))

    ;; reset gradient
    (set-value-grad! p 0.0)))

;; helpers for kv-cache
(define (zero-embedding)
  (for/vector ([i n-embd])
    (make-value 0.0)))

(define (make-kv-cache)
  (for/vector ([layer n-layer])
    (for/vector ([pos block-size])
      (zero-embedding))))

(define (train-step step doc)
  ;; tokenization
  (define tokens
    (list->vector
     (append
      (list BOS)
      (map char->token (string->list doc))
      (list BOS))))

  (define n (min block-size (- (vector-length tokens) 1)))
  ;; kv cache
  (define keys (make-kv-cache))
  (define values (make-kv-cache))
  
  (define losses (make-vector n (make-value 0.0)))
  (for ([pos (in-range n)])
    (define token-id (vector-ref tokens pos))
    (define target-id (vector-ref tokens (+ pos 1)))

    (define logits (gpt token-id pos keys values))
    (define probs (softmax logits))

    (define loss
      (vneg (vlog (vector-ref probs target-id))))

    (vector-set! losses pos loss))
  
  (define loss
    (v* (make-value (/ 1 n))
        (let loop ((i 0)
                   (acc (make-value 0.0)))
          (if (= i (vector-length losses))
              acc
              (loop (+ i 1)
                    (v+ acc (vector-ref losses i)))))))
  
  (backward loss)
  
  (adam-step param-list-v m v step learning-rate beta1 beta2 eps-adam)
  
  loss)

(define (train-loop)
  (let loop ((step 0))
    (when (< step num-steps)
      (define doc (vector-ref docs-vec
                              (modulo step (vector-length docs-vec))))

      (define loss (train-step step doc))
      
      (displayln
       (format "step ~a / ~a | loss ~a"
               (+ step 1)
               num-steps
               (value-data loss)))
      (loop (+ step 1))
      )))

;; inference
(define temperature 0.5)

;; helper
(define (weighted-random-choice weights)
  (define total
    (let loop ((i 0) (acc 0.0))
      (if (= i (vector-length weights))
          acc
          (loop (+ i 1)
                (+ acc (vector-ref weights i))))))
  (define r (* total (random)))
  (let loop ((i 0)
             (acc 0.0))
    (cond
      [(= i (vector-length weights))
       (- (vector-length weights) 1)]
      [else
       (define next (+ acc (vector-ref weights i)))
       (if (< r next)
           i
           (loop (+ i 1) next))])))

(define (sample-step pos token-id keys values)
  (define logits (gpt token-id pos keys values))
  ;; temperature scaling (scalar-safe)
  (define scaled
    (for/vector ([i (in-range (vector-length logits))])
      (v* (vector-ref logits i)
          (make-value (/ 1.0 temperature)))))
  (define probs (softmax scaled))

  (define weights
    (for/vector ([i (in-range (vector-length probs))])
      (value-data (vector-ref probs i))))

  (weighted-random-choice weights))

;; generation
(define (generate)
  (define keys (make-kv-cache))
  (define values (make-kv-cache))

  (define out '())
  (define token-id BOS)

  (let loop ((pos 0))
    (when (< pos (- block-size 1))

      (set! token-id
            (sample-step pos token-id keys values))

      (unless (= token-id BOS)
        (set! out (cons (vector-ref uchars token-id) out))
        (loop (+ pos 1)))))

  (list->string (reverse out)))

;; batch generation
(define (generate-n n)
  (for ([i (in-range n)])
    (printf "sample ~a: ~a\n"
            (+ i 1)
            (generate))))

;; save model
(define (save-model filename)
  (call-with-output-file filename
    #:exists 'replace
    (lambda (out)
      (write
       (map value-data (params state-dict))
       out))))

(define (load-model filename)
  (for ([p (params state-dict)]
        [v (call-with-input-file filename read)])
    (set-value-data! p v)))

(train-loop)

;;(save-model "model.dat")


;;(load-model "model.dat")
(generate-n 20)