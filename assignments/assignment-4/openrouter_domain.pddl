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
    (llm-has-capability ?l - llm ?c - capability)
    (llm-supported-by ?l - llm ?p - provider)
    (llm-has-cost ?l - llm ?tier - cost-tier)
    (llm-context-limit ?l - llm)
    (account-has-budget ?a - account)
    (request-needs-capability ?r - request ?c - capability)
    (request-assigned-to ?r - request ?l - llm)
  )

  (:action assign-request
    :parameters (?r - request ?l - llm ?c - capability)
    :precondition (and
      (request-needs-capability ?r ?c)
      (llm-has-capability ?l ?c)
    )
    :effect (request-assigned-to ?r ?l)
  )
)
