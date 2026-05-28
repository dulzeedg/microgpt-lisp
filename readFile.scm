#lang racket
;read file
(define names (list "abdul"))
(call-with-input-file "names.txt"
  (lambda (i)
    (define (readrec i)
      (let ((name (read-line i)))
        (if (eof-object? name)
            (display "END")
            (begin
              (set! names
                   (cons (string-trim name) names))
              (readrec i)))))
    (readrec i)))
(set! names (reverse names))
;(string-join names "")
(define uchars
  '("a" "b" "c" "d" "e" "f" "g"
    "h" "i" "j" "k" "l" "m"
    "n" "o" "p" "q" "r" "s"
    "t" "u" "v" "w" "x" "y" "z"))
(define BOS (length uchars))
(define vocab_size (+ (length uchars) 1))
(display vocab_size)