:- module(quantum_agent,
          [ permitted/1,
            prohibited_action/1,
            entropy_ok/1,
            symmetric_entry/4,
            proof_output/5,
            verify_or_refute/3
          ]).

% Host applications may assert prohibited_action/1 from their policy store.
:- dynamic prohibited_action/1.

% Safety invariant requested by the execution harness.
permitted(Action) :- \+ prohibited_action(Action).

entropy_ok(H) :- number(H), H >= 0.0, H =< 0.20.

% Q = (Q + Q^T) / 2, using zero-based conceptual indices represented as
% ordinary Prolog list positions (1-based) below.
symmetric_entry(Q, I, J, V) :-
    nth1(I, Q, RowI), nth1(J, RowI, A),
    nth1(J, Q, RowJ), nth1(I, RowJ, B),
    V is (A + B) / 2.0.

finite_number(X) :- number(X), X =:= X.

proof_output(Cycle, State, Entropy, Rule, proof(Cycle, Hash, Entropy, Rule, State)) :-
    integer(Cycle), Cycle >= 0,
    maplist(finite_number, State),
    entropy_ok(Entropy),
    atom(Rule),
    term_hash(state(Cycle, State), Hash).

same_trace(trace(Sid, D, State, Operator, H, Prev, Hash),
           trace(Sid, D, State, Operator, H, Prev, Hash)).

verify_or_refute(Action, Trace, committed) :-
    permitted(Action),
    Trace = trace(_, _, State, Operator, H, _, _),
    maplist(maplist(finite_number), Operator),
    maplist(finite_number, State),
    entropy_ok(H), !.
verify_or_refute(_, _, model_refuted).

% Example setup:
% ?- assertz(prohibited_action(delete)), permitted(delete).
% false.
% ?- proof_output(1, [0.1,0.2], 0.1, 'finite transition', P).
