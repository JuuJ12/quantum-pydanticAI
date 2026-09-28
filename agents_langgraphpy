from typing import Optional, TypedDict
from langgraph.graph import StateGraph, START, END
 
# Reaproveita exatamente as funções que já existem no seu agents_pydantic.py —
# as chains do LangChain (prompt | model.with_structured_output(...)) não mudam nada aqui.
from agents_pydantic import (
    agent_extrator,
    agent_builder_reponse,
    agent_verifier_plan,
    agent_executor_circuit,
    agent_metric,
    agent_verifier_execution,
    agent_synthesizer,
)
from agent_classes import StructuredCircuit, CircuitPlan, CircuitMetrics
 
 
class PipelineState(TypedDict, total=False):
    user_prompt: str
    max_attempts: int
    attempt: int
    requirements: StructuredCircuit
    plan: CircuitPlan
    qc: object
    measurement_counts: dict
    image_bytes: Optional[bytes]
    metrics: CircuitMetrics
    summary: str
    last_rejection_reason: Optional[str]
    success: bool
 
 
# ---------------------------------------------------------------------------
# Nós — cada um só chama a função de agente que você já tinha
# ---------------------------------------------------------------------------
def node_extract(state: PipelineState) -> dict:
    requirements = agent_extrator(state["user_prompt"])
    return {"requirements": requirements, "attempt": 0}
 
 
def node_build_plan(state: PipelineState) -> dict:
    requirements = state["requirements"]
    reason = state.get("last_rejection_reason")
    attempt = state.get("attempt", 0) + 1
 
    if reason:
        feedback_input = requirements.model_copy(update={
            "objective": f"{requirements.objective} [CORREÇÃO NECESSÁRIA - tentativa {attempt}: {reason}]"
        })
        plan = agent_builder_reponse(feedback_input)
    else:
        plan = agent_builder_reponse(requirements)
 
    return {"plan": plan, "attempt": attempt}
 
 
def node_verify_plan(state: PipelineState) -> dict:
    check = agent_verifier_plan(state["requirements"], state["plan"])
    if not check.approved:
        return {"last_rejection_reason": f"Plano inválido: {check.reason}"}
    return {"last_rejection_reason": None}
 
 
def node_execute(state: PipelineState) -> dict:
    try:
        qc, counts, image_bytes = agent_executor_circuit(state["plan"])
        return {
            "qc": qc,
            "measurement_counts": counts,
            "image_bytes": image_bytes,
            "last_rejection_reason": None,
        }
    except Exception as exc:
        return {"last_rejection_reason": f"Erro na execução do circuito: {exc}"}
 
 
def node_metrics(state: PipelineState) -> dict:
    metrics = agent_metric(state["qc"], state["requirements"], state["measurement_counts"])
    return {"metrics": metrics}
 
 
def node_verify_execution(state: PipelineState) -> dict:
    check = agent_verifier_execution(state["measurement_counts"], state["requirements"], state["metrics"])
    if not check.approved:
        return {"last_rejection_reason": f"Resultado inválido: {check.reason}"}
    return {"last_rejection_reason": None}
 
 
def node_synthesize(state: PipelineState) -> dict:
    summary = agent_synthesizer(
        requirements=state["requirements"].model_dump(),
        planning=state["plan"].model_dump(),
        metrics=state["metrics"].model_dump(),
    )
    return {"summary": summary, "success": True}
 
 
def node_fail(state: PipelineState) -> dict:
    return {"success": False}
 
 
# ---------------------------------------------------------------------------
# Roteamento condicional — substitui o "continue" do for loop original
# ---------------------------------------------------------------------------
def _route(state: PipelineState, next_step: str) -> str:
    if state.get("last_rejection_reason"):
        if state["attempt"] >= state["max_attempts"]:
            return "fail"
        return "retry"
    return next_step
 
 
def route_after_plan_check(state: PipelineState) -> str:
    return _route(state, "execute")
 
 
def route_after_execute(state: PipelineState) -> str:
    return _route(state, "metrics")
 
 
def route_after_exec_check(state: PipelineState) -> str:
    return _route(state, "synthesize")
 
 
# ---------------------------------------------------------------------------
# Montagem do grafo
# ---------------------------------------------------------------------------
builder = StateGraph(PipelineState)
builder.add_edge(START, "extract")
builder.add_node("extract", node_extract)
builder.add_node("build_plan", node_build_plan)
builder.add_node("verify_plan", node_verify_plan)
builder.add_node("execute", node_execute)
builder.add_node("metrics", node_metrics)
builder.add_node("verify_execution", node_verify_execution)
builder.add_node("synthesize", node_synthesize)
builder.add_node("fail", node_fail)

#aqui estamos criando arestas manualmente.
builder.add_edge("extract", "build_plan")
builder.add_edge("build_plan", "verify_plan")

#como ler os builders com condição: "Depois que o nó 'x' rodar, chame a função 'y' passando o estado atual, o que a função devolver vai ser usado como chave nesse dicionário que vou passar:
# {dicionário} e o valor da chave vai ser o próximo nó que o grafo vai chamar."
#De forma resumida, no final esse builder vai criar uma aresta para um nó com base na lógica que construimos as funções "router"
builder.add_conditional_edges("verify_plan", route_after_plan_check, {
    "execute": "execute",
    "retry": "build_plan",
    "fail": "fail",
})
builder.add_conditional_edges("execute", route_after_execute, {
    "metrics": "metrics",
    "retry": "build_plan",
    "fail": "fail",
})
 
builder.add_edge("metrics", "verify_execution")
 
builder.add_conditional_edges("verify_execution", route_after_exec_check, {
    "synthesize": "synthesize",
    "retry": "build_plan",
    "fail": "fail",
})
 
builder.add_edge("synthesize", END)
builder.add_edge("fail", END)
 
quantum_pipeline_graph = builder.compile()
 
 
def run_quantum_pipeline_graph(user_prompt: str, max_attempts: int = 3) -> dict:
    final_state = quantum_pipeline_graph.invoke({
        "user_prompt": user_prompt,
        "max_attempts": max_attempts,
    })
 
    if not final_state.get("success"):
        raise RuntimeError(
            f"Não foi possível construir um circuito válido após {max_attempts} tentativas. "
            f"Último problema: {final_state.get('last_rejection_reason')}"
        )
 
    return final_state

print(run_quantum_pipeline_graph("Crie um circuito de Bell com 2 qubits, medindo a fidelidade do estado final."))