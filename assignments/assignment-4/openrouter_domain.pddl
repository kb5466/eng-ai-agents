(define (domain openrouter)
  (:requirements :strips :typing)

  (:types
    llm
    provider
    capability
    request
    account
    cost-tier
  )

  (:predicates
    ;; Capabilities and support
    (llm-has-capability ?l - llm ?c - capability)
    (request-needs-capability ?r - request ?c - capability)

    ;; LLM to provider and cost
    (llm-supported-by ?l - llm ?p - provider)
    (llm-has-cost ?l - llm ?tier - cost-tier)

    ;; Budget (represented in tiers)
    (account-has-budget ?a - account ?tier - cost-tier)

    ;; Token limit (binary for simplicity)
    (llm-context-ok ?l - llm)

    ;; Routing
    (request-assigned-to ?r - request ?l - llm)
    (request-unassigned ?r - request)
  )

  (:action assign-request
  :parameters (?r - request ?l - llm ?c - capability)
  :precondition (and
    (request-unassigned ?r)
    (request-needs-capability ?r ?c)
    (llm-has-capability ?l ?c)
  )
  :effect (and
    (request-assigned-to ?r ?l)
    (not (request-unassigned ?r))
  )
)
)
