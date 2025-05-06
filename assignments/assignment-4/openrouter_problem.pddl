(define (problem route-requests)
  (:domain openrouter)

  (:objects
    ;; LLMs
    gpt4 mistral codegen - llm

    ;; Providers
    openai mistral-ai huggingface - provider

    ;; Capabilities
    code multilingual safe-for-kids - capability

    ;; Requests
    req1 req2 - request

    ;; Accounts
    account1 - account

    ;; Cost tiers
    low-cost medium-cost high-cost - cost-tier
  )

  (:init
    ;; LLM capabilities
    (llm-has-capability gpt4 code)
    (llm-has-capability gpt4 multilingual)
    (llm-has-capability mistral code)
    (llm-has-capability codegen code)
    (llm-has-capability codegen safe-for-kids)

    ;; LLM providers
    (llm-supported-by gpt4 openai)
    (llm-supported-by mistral mistral-ai)
    (llm-supported-by codegen huggingface)

    ;; LLM costs
    (llm-has-cost gpt4 high-cost)
    (llm-has-cost mistral medium-cost)
    (llm-has-cost codegen low-cost)

    ;; Token context OK (for all LLMs)
    (llm-context-ok gpt4)
    (llm-context-ok mistral)
    (llm-context-ok codegen)

    ;; Account budget
    (account-has-budget account1 high-cost)

    ;; Requests and their required capabilities
    (request-needs-capability req1 code)
    (request-needs-capability req2 multilingual)

    (request-unassigned req1)
    (request-unassigned req2)
  )

  (:goal
    (and
      (request-assigned-to req1 mistral)
      (request-assigned-to req2 gpt4)
    )
  )
)
