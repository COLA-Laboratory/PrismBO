from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Dict, Any


Token = Tuple[str, str]  # (type, value)

TAG_OPEN_RE   = re.compile(r'<([a-zA-Z][a-zA-Z0-9_]*)>')
TAG_CLOSE_RE  = re.compile(r'</([a-zA-Z][a-zA-Z0-9_]*)>')
STRING_RE     = re.compile(r'"([A-Za-z0-9_ .,\\-]*)"')
NUMBER_RE     = re.compile(r'\d+(?:\.\d+)?')
WORD_RE       = re.compile(r'[A-Za-z]+')


KEYWORDS = {
    'minimize', 'maximize',
    'real', 'integer', 'categorical',
    'eq', 'ineq',
}

SYMBOLS = {
    '[', ']', '{', '}', ',', 
}

WHITESPACE_RE = re.compile(r'\s+')

def tokenize(s: str) -> List[Token]:
    i = 0
    n = len(s)
    tokens: List[Token] = []
    while i < n:
        # skip ws
        m = WHITESPACE_RE.match(s, i)
        if m:
            i = m.end()
            if i >= n:
                break
        if i >= n:
            break

        # tags
        m = TAG_OPEN_RE.match(s, i)
        if m:
            tokens.append(('TAG_OPEN', m.group(1)))
            i = m.end()
            continue
        m = TAG_CLOSE_RE.match(s, i)
        if m:
            tokens.append(('TAG_CLOSE', m.group(1)))
            i = m.end()
            continue

        # symbols
        if s[i] in SYMBOLS:
            tokens.append((s[i], s[i]))
            i += 1
            continue

        # comma
        if s[i] == ',':
            tokens.append((',', ','))
            i += 1
            continue

        # string
        m = STRING_RE.match(s, i)
        if m:
            tokens.append(('STRING', m.group(1)))
            i = m.end()
            continue

        # number
        m = NUMBER_RE.match(s, i)
        if m:
            tokens.append(('NUMBER', m.group(0)))
            i = m.end()
            continue

        # keywords/words
        m = WORD_RE.match(s, i)
        if m:
            word = m.group(0)
            tok_type = 'KEYWORD' if word in KEYWORDS else 'IDENT'
            tokens.append((tok_type, word))
            i = m.end()
            continue

        raise SyntaxError(f'Unexpected character at {i}: {s[i]!r}')
    return tokens


@dataclass
class Objective:
    name: str
    direction: str  # 'minimize' | 'maximize'

@dataclass
class VarRangeRealInt:
    low: float
    high: float

@dataclass
class VarRangeCategorical:
    choices: List[str]

VarRange = Union[VarRangeRealInt, VarRangeCategorical]

@dataclass
class Variable:
    name: str
    vtype: str     # 'real' | 'integer' | 'categorical'
    vrange: VarRange

@dataclass
class Constraint:
    name: str
    ctype: str     # 'eq' | 'ineq'

@dataclass
class Statistics:
    data_num: Optional[int] = None
    mean: Optional[float] = None
    variance: Optional[float] = None

@dataclass
class Task:
    name: str
    desc: str
    objectives: List[Objective]
    variables: List[Variable]
    constraints: List[Constraint]
    statistics: Optional[Statistics]


class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def eat(self, typ: str, val: Optional[str]=None) -> Token:
        tok = self.peek()
        if tok is None:
            raise SyntaxError(f'Expected {typ}{" "+val if val else ""}, got EOF')
        if tok[0] != typ or (val is not None and tok[1] != val):
            raise SyntaxError(f'Expected {typ}{" "+val if val else ""}, got {tok}')
        self.pos += 1
        return tok

    def try_eat(self, typ: str, val: Optional[str]=None) -> Optional[Token]:
        tok = self.peek()
        if tok and tok[0] == typ and (val is None or tok[1] == val):
            self.pos += 1
            return tok
        return None

    # <task> ::= "<task>" <name> <description> <objectives> <variables> [<constraints>] [<statistics>] "</task>"
    def parse_task(self) -> Task:
        self.eat('TAG_OPEN', 'task')
        name = self.parse_name()
        desc = self.parse_desc()
        objectives = self.parse_objectives()
        variables = self.parse_variables()
        constraints: List[Constraint] = []
        statistics: Optional[Statistics] = None

        # optional <constraints>
        if self._next_is_tag_open('constraints'):
            constraints = self.parse_constraints()

        # optional <stats>
        if self._next_is_tag_open('stats'):
            statistics = self.parse_stats()

        self.eat('TAG_CLOSE', 'task')
        return Task(
            name=name,
            desc=desc,
            objectives=objectives,
            variables=variables,
            constraints=constraints,
            statistics=statistics,
        )

    # <name> ::= "<name>" <string> "</name>"
    def parse_name(self) -> str:
        self.eat('TAG_OPEN', 'name')
        s = self.eat('STRING')[1]
        self.eat('TAG_CLOSE', 'name')
        return s

    # <description> ::= "<desc>" <string> "</desc>"
    def parse_desc(self) -> str:
        self.eat('TAG_OPEN', 'desc')
        s = self.eat('STRING')[1]
        self.eat('TAG_CLOSE', 'desc')
        return s

    # <objectives> ::= "<objectives>" { <objective> } "</objectives>"
    # <objective> ::= "<obj>" <obj_name> <direction> "</obj>"
    def parse_objectives(self) -> List[Objective]:
        self.eat('TAG_OPEN', 'objectives')
        objs: List[Objective] = []
        while self._next_is_tag_open('obj'):
            objs.append(self.parse_objective())
        self.eat('TAG_CLOSE', 'objectives')
        if not objs:
            raise SyntaxError('At least one <obj> required inside <objectives>.')
        return objs

    def parse_objective(self) -> Objective:
        self.eat('TAG_OPEN', 'obj')
        # <obj_name> ::= <string>
        name = self.eat('STRING')[1]
        # <direction> ::= "minimize" | "maximize"
        dir_tok = self.eat('KEYWORD')
        if dir_tok[1] not in ('minimize', 'maximize'):
            raise SyntaxError('Objective direction must be "minimize" or "maximize".')
        self.eat('TAG_CLOSE', 'obj')
        return Objective(name=name, direction=dir_tok[1])

    # <variables> ::= "<variables>" { <variable> } "</variables>"
    # <variable> ::= "<var>" <var_name> <var_type> <range> "</var>"
    def parse_variables(self) -> List[Variable]:
        self.eat('TAG_OPEN', 'variables')
        vars: List[Variable] = []
        while self._next_is_tag_open('var'):
            vars.append(self.parse_variable())
        self.eat('TAG_CLOSE', 'variables')
        if not vars:
            raise SyntaxError('At least one <var> required inside <variables>.')
        return vars

    def parse_variable(self) -> Variable:
        self.eat('TAG_OPEN', 'var')
        name = self.eat('STRING')[1]
        vtype_tok = self.eat('KEYWORD')
        if vtype_tok[1] not in ('real', 'integer', 'categorical'):
            raise SyntaxError('var_type must be real|integer|categorical')
        vtype = vtype_tok[1]
        vrange = self.parse_range(vtype)
        self.eat('TAG_CLOSE', 'var')
        return Variable(name=name, vtype=vtype, vrange=vrange)

    def parse_range(self, vtype: str) -> VarRange:
        if self.try_eat('['):
            low = float(self.eat('NUMBER')[1])
            self.eat(',', ',')
            high = float(self.eat('NUMBER')[1])
            self.eat(']', ']')
            if vtype == 'integer':
                # keep as floats in AST, but we can validate ordering and integer-ness
                if not low.is_integer() or not high.is_integer():
                    raise SyntaxError('Integer variable range bounds must be integers.')
            if high < low:
                raise SyntaxError('Range upper bound must be >= lower bound.')
            return VarRangeRealInt(low=low, high=high)
        elif self.try_eat('{'):
            choices: List[str] = []
            choices.append(self.eat('STRING')[1])
            while self.try_eat(',', ','):
                choices.append(self.eat('STRING')[1])
            self.eat('}', '}')
            if vtype != 'categorical':
                raise SyntaxError('Only categorical variables may use {choice,...} ranges.')
            if len(choices) == 0:
                raise SyntaxError('Categorical list must contain at least one string.')
            return VarRangeCategorical(choices=choices)
        else:
            raise SyntaxError('Expected a range: [low,high] or {"a","b",...}')

    # <constraints> ::= "<constraints>" { <constraint> } "</constraints>"
    # <constraint> ::= "<con>" <con_name> <con_type> "</con>"
    def parse_constraints(self) -> List[Constraint]:
        self.eat('TAG_OPEN', 'constraints')
        cons: List[Constraint] = []
        while self._next_is_tag_open('con'):
            self.eat('TAG_OPEN', 'con')
            name = self.eat('STRING')[1]
            ctype_tok = self.eat('KEYWORD')
            if ctype_tok[1] not in ('eq', 'ineq'):
                raise SyntaxError('Constraint type must be eq|ineq')
            self.eat('TAG_CLOSE', 'con')
            cons.append(Constraint(name=name, ctype=ctype_tok[1]))
        self.eat('TAG_CLOSE', 'constraints')
        return cons

    # <statistics> ::= "<stats>" [data_num] [<mean>] [<variance>] "</stats>"
    # <data_num> ::= "<dnum>" <num> "</dnum>"
    # <mean> ::= "<mean>" <number> "</mean>"
    # <variance> ::= "<variance>" <number> "</variance>"
    def parse_stats(self) -> Statistics:
        self.eat('TAG_OPEN', 'stats')
        dnum: Optional[int] = None
        mean: Optional[float] = None
        var: Optional[float] = None

        if self._next_is_tag_open('dnum'):
            self.eat('TAG_OPEN', 'dnum')
            dnum = int(self.eat('NUMBER')[1])
            self.eat('TAG_CLOSE', 'dnum')

        if self._next_is_tag_open('mean'):
            self.eat('TAG_OPEN', 'mean')
            mean = float(self.eat('NUMBER')[1])
            self.eat('TAG_CLOSE', 'mean')

        if self._next_is_tag_open('variance'):
            self.eat('TAG_OPEN', 'variance')
            var = float(self.eat('NUMBER')[1])
            self.eat('TAG_CLOSE', 'variance')

        self.eat('TAG_CLOSE', 'stats')
        return Statistics(data_num=dnum, mean=mean, variance=var)

    # helpers
    def _next_is_tag_open(self, name: str) -> bool:
        tok = self.peek()
        return bool(tok and tok[0] == 'TAG_OPEN' and tok[1] == name)

def parse_task_from_string(s: str) -> Task:
    if len(s) == 0:
        return ""
    toks = tokenize(s)
    parser = Parser(toks)
    task = parser.parse_task()
    # ensure all tokens were consumed
    if parser.peek() is not None:
        raise SyntaxError(f'Unexpected trailing tokens starting at: {parser.peek()}')
    return task


def task_to_dict(task: Task) -> Dict[str, Any]:
    def vr_to_obj(vr: VarRange) -> Dict[str, Any]:
        if isinstance(vr, VarRangeRealInt):
            return {"type": "range", "low": vr.low, "high": vr.high}
        else:
            return {"type": "categorical", "choices": list(vr.choices)}
    return {
        "name": task.name,
        "description": task.desc,
        "objectives": [{"name": o.name, "direction": o.direction} for o in task.objectives],
        "variables": [{
            "name": v.name,
            "type": v.vtype,
            "range": vr_to_obj(v.vrange)
        } for v in task.variables],
        "constraints": [{"name": c.name, "type": c.ctype} for c in task.constraints],
        "statistics": (
            None if task.statistics is None else
            {"data_num": task.statistics.data_num,
             "mean": task.statistics.mean,
             "variance": task.statistics.variance}
        )
    }


if __name__ == "__main__":
    example = """
    <task>
    <name>"Speed Reducer Design"</name>
    <desc>"A mechanical design task with one objective and inequality constraints on strength and size."</desc>

    <objectives>
        <obj>"weight" minimize</obj>
    </objectives>

    <variables>
        <var>"face_width" real [2.6, 3.6]</var>
        <var>"teeth_number" integer [17, 28]</var>
        <var>"material_type" categorical {"steel", "aluminum", "titanium"}</var>
    </variables>

    <constraints>
        <con>"stress_limit" ineq</con>
        <con>"gear_ratio_fix" eq</con>
    </constraints>

    <stats>
        <mean>295.3</mean>
        <variance>25.1</variance>
    </stats>
    </task>
    """
    task = parse_task_from_string(example)
    import json
    print(json.dumps(task_to_dict(task), indent=2))