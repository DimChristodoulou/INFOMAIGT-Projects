import random


class LSystem:
    def __init__(self, axiom, rules):
        self.axiom = axiom
        # rules_obj = list
        #
        # for rule in rules:
        #     if rule[0] == 'S':
        #         rule_obj = StochasticRule(rule)
        #     else:
        #         rule_obj = Rule(rule)
        #     rules_obj.append(rule_obj)
        #
        # self.rules = rules_obj
        self.rules = rules

    def evaluate(self, iterations):
        current = self.axiom
        for i in range(iterations):
            new = []
            for c in current:
                if c in self.rules:
                    new.append(str(self.rules[c]))
                else:
                    new.append(c)

            current = ''.join(new)

        return current


class Rule:
    def __init__(self, rule):
        self.rule = rule

    def __str__(self):
        return self.rule


class StochasticRule(Rule):
    def __str__(self):
        rng = random.random()
        for p, r in self.rule:
            if rng <= p:
                return r
            rng -= p
