#lang racket
; Autograd engine for AI
; A Value Node
; data        -> scalar number
; grad        -> accumulated gradient
; children    -> parent nodes
; local-grads -> local derivatives

(define (make-value data)
  (vector data ; data
          0.0  ; grad  
          '()  ; children
          '())) ; local-grads

;; Accessors
(define (value-data v)
  (vector-ref v 0))

(define (value-grad v)
  (vector-ref v 1))

(define (value-children v)
  (vector-ref v 2))

(define (value-local-grads v)
  (vector-ref v 3))

;; Mutators
(define (set-value-grad! v x)
  (vector-set! v 1 x))

(define (set-value-children! v xs)
  (vector-set! v 2 xs))

(define (set-value-local-grads! v xs)
  (vector-set! v 3 xs))

;; Operations
(define (v+ a b)
  (let ((out (make-value (+
                          (value-data a)
                          (value-data b)))))
    (set-value-children! out (list a b))
    (set-value-local-grads! out '(1.0 1.0))
    out))

(define (v* a b)
  (let ((out (make-value (*
                          (value-data a)
                          (value-data b)))))
    (set-value-children! out (list a b))
    (set-value-local-grads! out (list (value-data b)
                                      (value-data a)))
    out))

(define (vpow a n)
  (let ((out (make-value (expt
                          (value-data a) n))))
    (set-value-children! out (list a))
    (set-value-local-grads! out (list (* n
                                         (expt (value-data a)
                                             (- n 1)))))
    out))

(define (vlog a)
  (let ((x (value-data a)))
    (let ((out (make-value (log x))))
      (set-value-children! out (list a))
      (set-value-local-grads!
       out
       (list (/ 1.0 x)))
      out)))

(define (vexp a)

  (let* ((x (value-data a))
         (e (exp x))
         (out (make-value e)))

    (set-value-children! out (list a))

    (set-value-local-grads!
     out
     (list e))

    out))

(define (vneg a)
  (v* a (make-value -1.0)))

(define (v- a b)
  (v+ a (vneg b)))

(define (v/ a b)
  (v* a (vpow b -1)))

;; RELU
(define (vrelu a)
  (let* ((x (value-data a))
        (out (make-value (if (> x 0) x 0))))
    (set-value-children! out (list a))
    (set-value-local-grads! out (list (if (> x 0)
                                          1.0
                                          0.0)))
    out))

;; Topological sort
(define (contains? xs x)
  (cond ((null? xs) #f)
        ((eq? (car xs) x) #t)
        (else (contains? (cdr xs) x))))

(define (build-topo v visited topo)
  (if (contains? visited v)
      (cons visited topo)
      (let ((visited2 (cons v visited)))
        (let loop ((children (value-children v))
                   (vis visited2)
                   (tp topo))
          (if (null? children)
              (cons vis (cons v tp))
              (let* ((result (build-topo (car children)
                                         vis
                                         tp))
                     (new-vis (car result))
                     (new-topo (cdr result)))
                (loop (cdr children)
                      new-vis
                      new-topo)))))))

;; Back propagation
(define (backward loss)

  ;; Build topo order
  (let* ((result (build-topo loss '() '()))
         (topo (cdr result)))

    ;; seed gradient
    (set-value-grad! loss 1.0)

    ;; reverse traversal
    (for-each
     (lambda (v)
       (let ((grad (value-grad v))
             (children (value-children v))
             (locals (value-local-grads v)))
         (let loop ((cs children)
                    (ls locals))
           (if (not (null? cs))
               (begin
                 (let* ((child (car cs))
                        (local-grad (car ls))
                        (new-grad
                         (+ (value-grad child)
                            (* local-grad grad))))
                   (set-value-grad! child new-grad)
                   (loop (cdr cs)
                         (cdr ls))))
               '()))))
     topo)))

;; ============================================================
;; TESTS
;; ============================================================

(displayln "====================================")
(displayln "TEST 1 : y = x^2 + 2x + 1")
(displayln "====================================")

(define x1 (make-value 3.0))

(define y1
  (v+
   (v+
    (vpow x1 2)
    (v* (make-value 2.0) x1))
   (make-value 1.0)))

(backward y1)

(display "y = ")
(displayln (value-data y1))

(display "dy/dx = ")
(displayln (value-grad x1))

(newline)

;; ============================================================

(displayln "====================================")
(displayln "TEST 2 : y = x + x")
(displayln "====================================")

(define x2 (make-value 5.0))
(define y2 (v+ x2 x2))

(backward y2)

(display "y = ")
(displayln (value-data y2))

(display "dy/dx = ")
(displayln (value-grad x2))

(newline)

;; ============================================================

(displayln "====================================")
(displayln "TEST 3 : y = x * x")
(displayln "====================================")

(define x3 (make-value 4.0))
(define y3 (v* x3 x3))

(backward y3)

(display "y = ")
(displayln (value-data y3))

(display "dy/dx = ")
(displayln (value-grad x3))

(newline)

;; ============================================================

(displayln "====================================")
(displayln "TEST 4 : ReLU positive")
(displayln "====================================")

(define x4 (make-value 3.0))
(define y4 (vrelu x4))

(backward y4)

(display "y = ")
(displayln (value-data y4))

(display "dy/dx = ")
(displayln (value-grad x4))

(newline)

;; ============================================================

(displayln "====================================")
(displayln "TEST 5 : ReLU negative")
(displayln "====================================")

(define x5 (make-value -3.0))
(define y5 (vrelu x5))

(backward y5)

(display "y = ")
(displayln (value-data y5))

(display "dy/dx = ")
(displayln (value-grad x5))

(newline)

;; ============================================================

(displayln "====================================")
(displayln "TEST 6 : y = x*x + x")
(displayln "====================================")

(define x6 (make-value 3.0))

(define y6
  (v+
   (v* x6 x6)
   x6))

(backward y6)

(display "y = ")
(displayln (value-data y6))

(display "dy/dx = ")
(displayln (value-grad x6))

(newline)

;; ============================================================

(displayln "====================================")
(displayln "TEST 7 : Deep chain")
(displayln "====================================")

(define x7 (make-value 2.0))

(define y7
  (vpow
   (vpow
    (vpow x7 2)
    2)
   2))

(backward y7)

(display "y = ")
(displayln (value-data y7))

(display "dy/dx = ")
(displayln (value-grad x7))

(newline)

;; ============================================================

(displayln "====================================")
(displayln "TEST 8 : Tiny neuron / AI")
(displayln "====================================")

(define w8 (make-value 2.0))
(define x8 (make-value 3.0))
(define b8 (make-value -1.0))

(define y8
  (vrelu
   (v+
    (v* w8 x8)
    b8)))

(backward y8)

(display "y = ")
(displayln (value-data y8))

(display "dy/dw = ")
(displayln (value-grad w8))

(display "dy/dx = ")
(displayln (value-grad x8))

(display "dy/db = ")
(displayln (value-grad b8))

(newline)

;; ============================================================

(displayln "====================================")
(displayln "TEST 9 : Branching graph")
(displayln "====================================")

(define x9 (make-value 2.0))

(define a9 (v* x9 x9))

(define b9
  (v+
   (v+ a9 a9)
   a9))

(define y9
  (v* b9 x9))

(backward y9)

(display "y = ")
(displayln (value-data y9))

(display "dy/dx = ")
(displayln (value-grad x9))

(newline)

(displayln "====================================")
(displayln "DONE")
(displayln "====================================")