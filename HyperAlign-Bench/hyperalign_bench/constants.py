from __future__ import annotations

DEFAULT_HYPERGRAPH_TOKEN = "<hypergraph>"

ARXIV_LABEL_TEXTS = [
    "cs.NA(Numerical Analysis)",
    "cs.MM(Multimedia)",
    "cs.LO(Logic in Computer Science)",
    "cs.CY(Computers and Society)",
    "cs.CR(Cryptography and Security)",
    "cs.DC(Distributed, Parallel, and Cluster Computing)",
    "cs.HC(Human-Computer Interaction)",
    "cs.CE(Computational Engineering, Finance, and Science)",
    "cs.NI(Networking and Internet Architecture)",
    "cs.CC(Computational Complexity)",
    "cs.AI(Artificial Intelligence)",
    "cs.MA(Multiagent Systems)",
    "cs.GL(General Literature)",
    "cs.NE(Neural and Evolutionary Computing)",
    "cs.SC(Symbolic Computation)",
    "cs.AR(Hardware Architecture)",
    "cs.CV(Computer Vision and Pattern Recognition)",
    "cs.GR(Graphics)",
    "cs.ET(Emerging Technologies)",
    "cs.SY(Systems and Control)",
    "cs.CG(Computational Geometry)",
    "cs.OH(Other Computer Science)",
    "cs.PL(Programming Languages)",
    "cs.SE(Software Engineering)",
    "cs.LG(Machine Learning)",
    "cs.SD(Sound)",
    "cs.SI(Social and Information Networks)",
    "cs.RO(Robotics)",
    "cs.IT(Information Theory)",
    "cs.PF(Performance)",
    "cs.CL(Computation and Language)",
    "cs.IR(Information Retrieval)",
    "cs.MS(Mathematical Software)",
    "cs.FL(Formal Languages and Automata Theory)",
    "cs.DS(Data Structures and Algorithms)",
    "cs.OS(Operating Systems)",
    "cs.GT(Computer Science and Game Theory)",
    "cs.DB(Databases)",
    "cs.DL(Digital Libraries)",
    "cs.DM(Discrete Mathematics)",
]


def build_nc_prompt() -> str:
    labels = ", ".join(ARXIV_LABEL_TEXTS)
    return (
        f"Given a node-centered hypergraph: {DEFAULT_HYPERGRAPH_TOKEN}, "
        "where nodes represent papers and hyperedges represent sets of papers "
        "co-cited by a source paper, please tell me which class the center node "
        f"belongs to. The 40 classes are: {labels}. Directly output the class name."
    )


def build_hecls_prompt() -> str:
    labels = ", ".join(ARXIV_LABEL_TEXTS)
    return (
        f"Given a hyperedge-centered hypergraph: {DEFAULT_HYPERGRAPH_TOKEN}, "
        "where the center hyperedge is induced by one source paper and its member "
        "nodes are the cited papers, please tell me which class the source paper "
        f"belongs to. The 40 classes are: {labels}. Directly output the class name."
    )


def build_nd_prompt() -> str:
    return f"Please briefly describe the center node of {DEFAULT_HYPERGRAPH_TOKEN}."


def build_hed_prompt() -> str:
    return f"Please briefly describe the center hyperedge of {DEFAULT_HYPERGRAPH_TOKEN}."


def build_nd_target(label_text: str, title: str | None = None) -> str:
    title = (title or "").strip()
    if title:
        return f'This is a paper in {label_text} domain, it\'s about "{title}".'
    return f"This is a paper in {label_text} domain."


def build_hed_target(label_text: str, title: str | None = None) -> str:
    title = (title or "").strip()
    if title:
        return (
            "This hyperedge is a co-citation group induced by a source paper in "
            f'{label_text} domain, associated with "{title}".'
        )
    return f"This hyperedge is a co-citation group induced by a source paper in {label_text} domain."
