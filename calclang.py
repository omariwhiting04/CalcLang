import sys
import re

# ---------------------------
# TOKEN DEFINITIONS
# ---------------------------

TOKENS = [
    ("NUMBER", r"\d+(\.\d+)?"),
    ("ID", r"[A-Za-z_]\w*"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MUL", r"\*"),
    ("DIV", r"/"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("ASSIGN", r"="),
    ("EQ", r"=="),
    ("NE", r"!="),
    ("LE", r"<="),
    ("GE", r">="),
    ("LT", r"<"),
    ("GT", r">"),
    ("COMMA", r","),
]

token_regex = [(typ, re.compile(pattern)) for typ, pattern in TOKENS]

KEYWORDS = {
    "if", "then", "else", "end", "do", "def", "while",
    "return", "print"
}

# ---------------------------
# LEXER


class Lexer:
    def __init__(self, code):
        self.code = code
        self.pos = 0
        self.tokens = []

    def tokenize(self):
        while self.pos < len(self.code):
            if self.code[self.pos].isspace():
                self.pos += 1
                continue

            match = None
            for tok_type, pattern in token_regex:
                match = pattern.match(self.code, self.pos)
                if match:
                    value = match.group()
                    if tok_type == "ID" and value in KEYWORDS:
                        self.tokens.append(("KEYWORD", value))
                    else:
                        self.tokens.append((tok_type, value))
                    self.pos = match.end()
                    break

            if not match:
                raise SyntaxError(f"Unexpected character: {self.code[self.pos]}")

        self.tokens.append(("EOF", None))
        return self.tokens



class Number:
    def __init__(self, value):
        self.value = float(value)

class Variable:
    def __init__(self, name):
        self.name = name

class BinOp:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class Assign:
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

class IfNode:
    def __init__(self, cond, then_body, else_body):
        self.cond = cond
        self.then_body = then_body
        self.else_body = else_body

class WhileNode:
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body

class FuncDef:
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class FuncCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args

class ReturnNode:
    def __init__(self, expr):
        self.expr = expr

class PrintNode:
    def __init__(self, expr):
        self.expr = expr

# ---------------------------
# PARSER

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current(self):
        return self.tokens[self.pos]

    def eat(self, expected):
        tok_type, tok_val = self.current()
        if tok_val == expected or tok_type == expected:
            self.pos += 1
            return tok_val
        raise SyntaxError(f"Expected {expected}, got {self.current()}")

    def parse(self):
        statements = []
        while self.current()[0] != "EOF":
            statements.append(self.parse_statement())
        return statements

    def parse_statement(self):
        tok_type, tok_val = self.current()

        if tok_val == "if":
            return self.parse_if()

        if tok_val == "while":
            return self.parse_while()

        if tok_val == "return":
            self.eat("return")
            expr = self.parse_expression()
            return ReturnNode(expr)

        if tok_val == "def":
            return self.parse_function()

        if tok_val == "print":
            self.eat("print")
            expr = self.parse_expression()
            return PrintNode(expr)

        if tok_type == "ID":
            name = self.eat("ID")
            if self.current()[0] == "ASSIGN":
                self.eat("ASSIGN")
                expr = self.parse_expression()
                return Assign(name, expr)
            else:
                return FuncCall(name, self.parse_arguments())

        raise SyntaxError(f"Unexpected statement: {self.current()}")

    # -------- BLOCK PARSERS ---------

    def parse_if(self):
        self.eat("if")
        cond = self.parse_expression()
        self.eat("then")

        then_body = []
        while not (self.current()[0] == "KEYWORD" and self.current()[1] in ("else", "end")):
            then_body.append(self.parse_statement())

        if self.current()[1] == "else":
            self.eat("else")
            else_body = []
            while not (self.current()[0] == "KEYWORD" and self.current()[1] == "end"):
                else_body.append(self.parse_statement())
            self.eat("end")
            return IfNode(cond, then_body, else_body)
        else:
            self.eat("end")
            return IfNode(cond, then_body, [])

    def parse_while(self):
        self.eat("while")
        cond = self.parse_expression()
        self.eat("do")

        body = []
        while not (self.current()[0] == "KEYWORD" and self.current()[1] == "end"):
            body.append(self.parse_statement())
        self.eat("end")
        return WhileNode(cond, body)

    def parse_function(self):
        self.eat("def")
        name = self.eat("ID")
        self.eat("LPAREN")
        params = []
        if self.current()[0] == "ID":
            params.append(self.eat("ID"))
            while self.current()[0] == "COMMA":
                self.eat("COMMA")
                params.append(self.eat("ID"))
        self.eat("RPAREN")

        body = []
        while not (self.current()[0] == "KEYWORD" and self.current()[1] == "end"):
            body.append(self.parse_statement())
        self.eat("end")

        return FuncDef(name, params, body)

    # -------- EXPRESSION PARSERS ---------

    def parse_expression(self):
        node = self.parse_addition()

        while self.current()[0] in ("LT", "GT", "LE", "GE", "EQ", "NE"):
            op = self.eat(self.current()[0])
            right = self.parse_addition()
            node = BinOp(node, op, right)

        return node

    def parse_addition(self):
        node = self.parse_term()

        while self.current()[0] in ("PLUS", "MINUS"):
            op = self.eat(self.current()[0])
            right = self.parse_term()
            node = BinOp(node, op, right)

        return node

    def parse_term(self):
        node = self.parse_factor()

        while self.current()[0] in ("MUL", "DIV"):
            op = self.eat(self.current()[0])
            right = self.parse_factor()
            node = BinOp(node, op, right)

        return node

    def parse_factor(self):
        tok_type, tok_val = self.current()

        if tok_type == "NUMBER":
            self.eat("NUMBER")
            return Number(tok_val)

        if tok_type == "ID":
            name = self.eat("ID")
            if self.current()[0] == "LPAREN":
                return FuncCall(name, self.parse_arguments())
            return Variable(name)

        if tok_type == "LPAREN":
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr

        raise SyntaxError("Invalid factor")

    def parse_arguments(self):
        args = []
        self.eat("LPAREN")
        if self.current()[0] != "RPAREN":
            args.append(self.parse_expression())
            while self.current()[0] == "COMMA":
                self.eat("COMMA")
                args.append(self.parse_expression())
        self.eat("RPAREN")
        return args



class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

class Interpreter:
    def __init__(self):
        self.globals = {}
        self.functions = {}
        self.call_stack = []

    def eval(self, node, env=None):
        if env is None:
            env = self.globals

        if isinstance(node, Number):
            return node.value

        if isinstance(node, Variable):
            if node.name in env:
                return env[node.name]
            raise RuntimeError(f"Undefined variable {node.name}")

        if isinstance(node, Assign):
            env[node.name] = self.eval(node.expr, env)
            return env[node.name]

        if isinstance(node, BinOp):
            left = self.eval(node.left, env)
            right = self.eval(node.right, env)

            if node.op == "+": return left + right
            if node.op == "-": return left - right
            if node.op == "*": return left * right
            if node.op == "/": return left / right

            if node.op == "<": return 1 if left < right else 0
            if node.op == ">": return 1 if left > right else 0
            if node.op == "<=": return 1 if left <= right else 0
            if node.op == ">=": return 1 if left >= right else 0
            if node.op == "==": return 1 if left == right else 0
            if node.op == "!=": return 1 if left != right else 0

            raise RuntimeError("Unknown operator")

        if isinstance(node, PrintNode):
            val = self.eval(node.expr, env)
            print(val)
            return val

        if isinstance(node, IfNode):
            if self.eval(node.cond, env) != 0:
                for stmt in node.then_body:
                    self.eval(stmt, env)
            else:
                for stmt in node.else_body:
                    self.eval(stmt, env)
            return None

        if isinstance(node, WhileNode):
            while self.eval(node.cond, env) != 0:
                for stmt in node.body:
                    self.eval(stmt, env)
            return None

        if isinstance(node, FuncDef):
            self.functions[node.name] = node
            return None

        if isinstance(node, FuncCall):
            if node.name not in self.functions:
                raise RuntimeError(f"Undefined function {node.name}")

            func = self.functions[node.name]
            args = [self.eval(a, env) for a in node.args]

            print(f"[CALL] {node.name}")
            self.call_stack.append(node.name)

            local_env = dict(env)
            for param, value in zip(func.params, args):
                local_env[param] = value

            try:
                for stmt in func.body:
                    self.eval(stmt, local_env)
            except ReturnException as r:
                self.call_stack.pop()
                print(f"[RETURN] {node.name}")
                return r.value

            self.call_stack.pop()
            print(f"[RETURN] {node.name}")
            return None

        if isinstance(node, ReturnNode):
            val = self.eval(node.expr, env)
            raise ReturnException(val)

        raise RuntimeError("Unknown AST node")

    def run(self, statements):
        for stmt in statements:
            self.eval(stmt)


def repl():
    interp = Interpreter()
    print("CalcLang REPL. Type 'end' alone on its own line to exit.")

    buffer = ""
    open_blocks = 0

    while True:
        line = input(">>> ")

        if line.strip() == "end" and buffer == "":
            break

        buffer += line + "\n"

        if line.strip().startswith(("if ", "while ", "def ")):
            open_blocks += 1

        if line.strip() == "end":
            open_blocks -= 1

        if open_blocks <= 0:
            try:
                lexer = Lexer(buffer)
                tokens = lexer.tokenize()
                parser = Parser(tokens)
                ast = parser.parse()
                interp.run(ast)
            except Exception as e:
                print("Error:", e)

            buffer = ""
            open_blocks = 0



if __name__ == "__main__":
    repl()
