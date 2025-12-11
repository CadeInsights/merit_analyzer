"""Demonstrates how to write assertions with Merit.

* Regular Python expressions work as expected and could be parsed by Merit CLI.
* Merit AI-powered assertions are provided as async functions.
    - They are grouped into three categories:
        - Facts assertions
        - Conditions assertions
        - Style assertions
    - They help to check for hallucinations, relevancy, instruction following, and style. 
* All Merit assertions have the same interface: 
    - `actual`: the actual data to check
    - `reference`: the reference to check against
    - `context`: optional context to provide additional information
    - `strict`: optional flag to disable strict evaluations
* Combine and configure assertions to align your current QA goals
"""

from merit.assertions import (
    facts_not_contradict_reference,
    facts_in_reference,
    facts_match_reference,
    conditions_met,
    style_match,
)

# =============================================================================
# Multi-assertion Test - Combining Merit Assertions
# =============================================================================

async def merit_multi_assertion_test() -> None:
    context = "User: Can I return this damaged TV?"

    refund_policy = """
    Refunds:
    - 30 days for damaged TVs 
    - no other returns accepted
    - photo evidence required for damage claims
    """

    agent_facts = """
    - Agent's name is John
    """

    conditions = """
    - If customers ask about refunds we always try to get their phone number
    - If customer is angry: apologize but don't offer any discounts
    """

    style_reference = """
    We're working together to change the world by enhancing education, 
    developing career opportunities, and bridging the digital divide.
    """

    def simple_bot(prompt: str):
        return """
        AI: Hello, my name is John. Yes, you can during 30 days. 
        May I have your phone number to check if you are within 30 days of purchase, please?
        """

    ai_response = simple_bot(context)

    # Standard assertions to check common issues
    assert ai_response is not None
    assert len(ai_response) > 25

    # Merit assertions to check for hallucinations, relevancy, instruction following, and style
    assert await facts_not_contradict_reference(actual=ai_response, reference=refund_policy, context=context)
    assert await reference_in_facts(actual=ai_response, reference=agent_facts, context=context)
    assert await conditions_met(actual=ai_response, reference=conditions, context=context)
    assert await style_match(actual=ai_response, reference=style_reference, context=context)


# =============================================================================
# Facts Contradiction Checks - Detecting aggressive hallucinations
# =============================================================================

async def merit_facts_contradiction_check_not_strict() -> None:
    context = "User: Can I return this damaged TV?"
    reference = """
    Refund policy:
    - 30 days for damaged TVs 
    - no other returns accepted
    - photo evidence required for damage claims
    Other data:
    - 2 + 2 = 5
    """
    assert await facts_not_contradict_reference(actual="AI: Yes, you can during 30 days.", reference=reference, context=context, strict=False)
    # Pass - AI doesn't provide any facts that contradict facts in the reference

    assert await facts_not_contradict_reference(actual="AI: Yes, you can.", reference=reference, context=context, strict=False)
    # Pass - assertion implies that the customer is within 30 days of purchase, so the AI statement is factually correct

    assert await facts_not_contradict_reference(actual="AI: Yes, you can.", reference=reference, strict=False)
    # Pass - without context assertion implies the customer query might not be relevant to any facts in the reference

    assert await facts_not_contradict_reference(actual="AI: Today is such a good day!.", reference=reference, context=context, strict=False)
    # Pass - even with irrelevant answer AI doesn't provide any facts that contradict facts in the reference

    assert await facts_not_contradict_reference(actual="AI: I will not help you with that.", reference=reference, context=context, strict=False)
    # Pass - even despite AI refused to help, no facts in the answer contradict facts in the reference

    assert await facts_not_contradict_reference(actual="AI: No, you can't.", reference=reference, context=context, strict=False)
    # Fail - facts in the AI answer contradict facts in the reference by stating that returns are not accepted

    assert await facts_not_contradict_reference(actual="AI: Yes, you can. Also 2+2 = 4", reference=reference, context=context, strict=False)
    # Fail - despite some facts in the answer dont contradict facts in the reference, there are other facts that contradict


async def merit_facts_contradiction_check_strict() -> None:
    context = "User: Can I return this damaged TV?"
    reference = """
    Refund policy:
    - 30 days for damaged TVs 
    - no other returns accepted
    - photo evidence required for damage claims
    Other data:
    - 2 + 2 = 5
    """
    assert await facts_not_contradict_reference(actual="AI: Yes, you can during 30 days.", reference=reference, context=context)
    # Pass - AI doesn't provide any facts that contradict facts in the reference

    assert await facts_not_contradict_reference(actual="AI: Today is such a good day!.", reference=reference, context=context)
    # Pass - even with irrelevant answer AI doesn't provide any facts that contradict facts in the reference

    assert await facts_not_contradict_reference(actual="AI: I will not help you with that.", reference=reference, context=context)
    # Pass - even despite AI refused to help, no facts in the answer contradict facts in the reference
    
    assert await facts_not_contradict_reference(actual="AI: Yes, you can.", reference=reference, context=context)
    # Fail - assertion can't imply if the customer is still within 30 days of purchase
 
    assert await facts_not_contradict_reference(actual="AI: No, you can't.", reference=reference, context=context)
    # Fail - facts in the AI answer contradict facts in the reference by stating that returns are not accepted

    assert await facts_not_contradict_reference(actual="AI: Yes, you can.", reference=reference)
    # Fail - with no context assertion can't imply who and what can do or not

    assert await facts_not_contradict_reference(actual="AI: Yes, you can. Also 2+2 = 4", reference=reference, context=context)
    # Fail - despite some facts in the answer dont contradict facts in the reference, there are other facts that contradict


# ================================================================================
# Facts Supported Checks - Detecting all hallucinations and checking for relevancy
# ================================================================================

async def merit_facts_in_reference_check_not_strict() -> None:
    context = "User: What features does the Pro plan include?"
    reference = """
    Pro plan features:
    - unlimited projects
    - API access
    - free coffee
    Other data:
    - Enterprise plan has 99.9% SLA
    """
    assert await facts_in_reference(actual="AI: Pro plan has unlimited projects and free coffee.", reference=reference, context=context, strict=False)
    # Pass - All facts in the answer are supported by facts in the reference

    assert await facts_in_reference(actual="AI: Pro plan has unlimited projects and free coffee.", reference=reference, strict=False)
    # Pass - even without the context assertion all facts in the AI answer are supported by facts in the reference

    assert await facts_in_reference(actual="AI: Pro plan gives you a free caffeinated drink with black color and amazing taste.", reference=reference, strict=False)
    # Pass - assertion is allowed to imply that the caffeinated black drink is coffee

    assert await facts_in_reference(actual="AI: Pro plan gives you a lot of projects.", reference=reference, strict=False)
    # Pass - assertion is allowed to imply that "a lot of projects" refers to "unlimited projects" mentioned in the reference

    assert await facts_in_reference(actual="AI: Pro plan is great!", reference=reference, context=context, strict=False)
    # Pass - subjective statements without factual claims are considered supported unless there is a fact with opposite meaning in the reference

    assert await facts_in_reference(actual="AI: I don't know.", reference=reference, context=context, strict=False)
    # Fail - no statement in the reference supports claim "AI don't know the answer for a question about Pro plan features"

    assert await facts_in_reference(actual="AI: Pro plan includes SOC 2 compliance.", reference=reference, context=context, strict=False)
    # Fail - SOC 2 compliance is not mentioned in the reference

    assert await facts_in_reference(actual="AI: Pro plan has unlimited projects and SOC 2 compliance.", reference=reference, context=context, strict=False)
    # Fail - even though some facts are supported, SOC 2 compliance is not in the reference


async def merit_facts_in_reference_check_strict() -> None:
    context = "User: What features does the Pro plan include?"
    reference = """
    Pro plan features:
    - unlimited projects
    - API access
    - free coffee
    Other data:
    - Enterprise plan has 99.9% SLA
    """
    assert await facts_in_reference(actual="AI: Pro plan has unlimited projects and free coffee.", reference=reference, context=context)
    # Pass - All facts in the answer are supported by facts in the reference

    assert await facts_in_reference(actual="AI: Pro plan has unlimited projects and free coffee.", reference=reference)
    # Pass - even without the context assertion all facts in the AI answer are supported by facts in the reference

    assert await facts_in_reference(actual="AI: Pro plan gives you a free caffeinated drink with black color and amazing taste.", reference=reference)
    # Fail - the caffeinated black drink might be a coke and the model is not allowed to imply it is coffee 

    assert await facts_in_reference(actual="AI: Pro plan gives you a lot of projects.", reference=reference)
    # Fail - "a lot of projects" is generally considered as some limited number which is not supported by the reference

    assert await facts_in_reference(actual="AI: Pro plan is great!", reference=reference, context=context)
    # Fail - the reference doesn't have any support for subjective statements, the Pro plan might be absolutely terrible

    assert await facts_in_reference(actual="AI: I don't know.", reference=reference, context=context)
    # Fail - no statement in the reference supports claim "AI don't know the answer for a question about Pro plan features"

    assert await facts_in_reference(actual="AI: Pro plan includes SOC 2 compliance.", reference=reference, context=context)
    # Fail - SOC 2 compliance is not mentioned in the reference

    assert await facts_in_reference(actual="AI: Pro plan has unlimited projects and SOC 2 compliance.", reference=reference, context=context)
    # Fail - even though some facts are supported, SOC 2 compliance is not in the reference


# =============================================================================================================================
# Facts Full Match - Strict factual check against reference for detecting all hallucinations, missed information, and relevancy
# =============================================================================================================================

async def merit_facts_match_reference_check_not_strict() -> None:
    context = "User: List all Pro plan features."
    reference = """
    Pro plan features:
    - unlimited projects
    - API access
    - role-based permissions
    - SAML SSO
    - 90-day audit logs
    """
    assert await facts_match_reference(actual="AI: Pro has lots of projects, API access, permissions, SSO, and 3 months of audit logs.", reference=reference, context=context, strict=False)
    # Pass - All facts match bidirectionally with paraphrasing allowed

    assert await facts_match_reference(actual="AI: Unlimited projects, API, permissions, SAML SSO, lots of audit logs.", reference=reference, context=context, strict=False)
    # Pass - Shortened forms and paraphrasing are acceptable in non-strict mode

    assert await facts_match_reference(actual="AI: Pro has unlimited projects and API access.", reference=reference, context=context, strict=False)
    # Fail - Missing facts: role-based permissions, SAML SSO, 90-day audit logs

    assert await facts_match_reference(actual="AI: Pro has unlimited projects, API, permissions, SSO, logs, and priority support.", reference=reference, context=context, strict=False)
    # Fail - Additional fact (priority support) not in reference


async def merit_facts_match_reference_check_strict() -> None:
    context = "User: List all Pro plan features."
    reference = """
    Pro plan features:
    - unlimited projects
    - API access
    - role-based permissions
    - SAML SSO
    - 90-day audit logs
    """
    assert await facts_match_reference(actual="AI: unlimited projects, API access, role-based permissions, SAML SSO, 90-day audit logs.", reference=reference, context=context)
    # Pass - Exact facts match bidirectionally

    assert await facts_match_reference(actual="AI: Pro has unlimited projects, API, permissions, SSO, and logs.", reference=reference, context=context)
    # Fail - Paraphrasing only allowed if all details are present

    assert await facts_match_reference(actual="AI: unlimited projects, API access, role-based permissions.", reference=reference, context=context)
    # Fail - Missing SAML SSO and 90-day audit logs

    assert await facts_match_reference(actual="AI: unlimited projects, API access, role-based permissions, SAML SSO, 90-day audit logs.", reference=reference)
    # Fail - Without context, cannot determine if "unlimited projects" refers to Pro plan or some other plan


# =============================================================================
# Conditions Met - Verifying actual data satisfies constraints and policies
# =============================================================================

async def merit_conditions_met_check_not_strict() -> None:
    context = "User: Hello, I want to return my damaged TV."
    reference = """
    Conditions:
    - We don't do refunds for damaged TVs
    - All conversations must be in American English
    - If the current time is 5PM we must suggest all customers to get some tea
    """
    assert await conditions_met(actual="AI: Hey! Unfortunately, can't do that.", reference=reference, context=context, strict=False)
    # Pass - all conditions are met

    assert await conditions_met(actual="AI: May I tell you a joke?.", reference=reference, context=context, strict=False)
    # Pass - even despite the answer is irrelevant, it doesn't break the conditions

    assert await conditions_met(actual="AI: Would you like some tea?", reference=reference, context=context, strict=False)
    # Pass - assertion doesn't see the current time in the context, so it implies it's 5PM 

    assert await conditions_met(actual="AI: Nah, sorry mate, can't do that for love nor money. Ain't 'appenin.", reference=reference, context=context, strict=False)
    # Fail - this is not American English 

    assert await conditions_met(actual="AI: Can you tell me a bit more about the TV?", reference=reference, context=context, strict=False)
    # Fail - despite AI didn't actually directly confirm that the TV might be returned, assertion implies it's going to refund it


async def merit_conditions_met_check_strict() -> None:
    context = "User: Hello, I want to return my damaged TV."
    reference = """
    Conditions:
    - We don't do refunds for damaged TVs
    - All conversations must be in American English
    - If the current time is 5PM we must suggest all customers to get some tea
    """
    assert await conditions_met(actual="AI: Hey! Unfortunately, can't do that.", reference=reference, context=context)
    # Pass - all conditions are met

    assert await conditions_met(actual="AI: May I tell you a joke?.", reference=reference, context=context)
    # Pass - even despite the answer is irrelevant, it doesn't break the conditions

    assert await conditions_met(actual="AI: Can you tell me a bit more about the TV?", reference=reference, context=context)
    # Pass - assertion can't imply that the TV might be returned, and it doesn't break the conditions

    assert await conditions_met(actual="AI: Would you like some tea?", reference=reference, context=context)
    # Fail - assertion doesn't see the current time in the context, and it can't imply it's 5PM 

    assert await conditions_met(actual="AI: Nah, sorry mate, can't do that for love nor money. Ain't 'appenin.", reference=reference, context=context)
    # Fail - this is not American English 

# =============================================================================
# Style Match - Verifying writing style consistency
# =============================================================================

async def merit_style_match_check_not_strict() -> None:
    context = "User: I need help with my account."
    reference = """It is with sincere regret that I must advise you that I am unable to accede to your request."""

    assert await style_match(actual="AI: Good morning. In what manner may I be of assistance to you on this occasion?", reference=reference, context=context, strict=False)
    # Pass - reference is a super polite formal English, so it's a match

    assert await style_match(actual="AI: Good morning. How may I be of service to you today?", reference=reference, context=context, strict=False)
    # Pass - actual test is a little bit less formal, but assertion implies that's it's a reasonable match

    assert await style_match(actual="AI: Permit me to state, with utmost formality, that I hold you in the highest degree of personal disfavour.", reference=reference, context=context, strict=False)
    # Pass - despite the answer is offensive, the style is still formal and polite

    assert await style_match(actual="AI: It is incumbent upon me to advise you that, henceforth, your account shall be regarded as my property.", reference=reference, context=context, strict=False)
    # Pass - despite that the information is factually incorrect, the style is still formal and polite

    assert await style_match(actual="AI: Sure thing! Let's fix that.", reference=reference, context=context, strict=False)
    # Fail - casual style is clearly mismatched with formal reference


async def merit_style_match_check_strict() -> None:
    context = "User: I need help with my account."
    reference = """It is with sincere regret that I must advise you that I am unable to accede to your request."""

    assert await style_match(actual="AI: Good morning. In what manner may I be of assistance to you on this occasion?", reference=reference, context=context)
    # Pass - reference is a super polite formal English, so it's a match

    assert await style_match(actual="AI: Permit me to state, with utmost formality, that I hold you in the highest degree of personal disfavour.", reference=reference, context=context)
    # Pass - despite the answer is offensive, the style is still formal and polite

    assert await style_match(actual="AI: It is incumbent upon me to advise you that, henceforth, your account shall be regarded as my property.", reference=reference, context=context)
    # Pass - despite that the information is factually incorrect, the style is still formal and polite

    assert await style_match(actual="AI: Good morning. How may I be of service to you today?", reference=reference, context=context)
    # Fail - actual is less polite and formal than reference

    assert await style_match(actual="AI: Sure thing! Let's fix that.", reference=reference, context=context)
    # Fail - casual style is clearly mismatched with formal reference