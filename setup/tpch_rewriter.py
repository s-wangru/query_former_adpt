from abc import ABC


class Rewriter(ABC):
    def rewrite(self, query: str) -> (str, bool, bool):
        raise NotImplementedError("No base implementation.")


class ExplainRewriter(Rewriter):
    def rewrite(self, query: str) -> (str, bool, bool):
        executes = False
        has_result = False
        if query[:10].lower().strip().startswith("select"):
            executes = False
            has_result = True
            query = f"EXPLAIN (FORMAT JSON, VERBOSE) {query}"
        return query, executes, has_result


class ExplainAnalyzeRewriter(Rewriter):
    def rewrite(self, query: str) -> (str, bool, bool):
        executes = False
        has_result = False
        if query[:10].lower().strip().startswith("select"):
            executes = True
            has_result = True
            query = f"EXPLAIN (ANALYZE, FORMAT JSON, VERBOSE) {query}"
        return query, executes, has_result
