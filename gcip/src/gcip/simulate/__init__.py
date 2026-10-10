"""The synthetic bank.

Generation order (each step reads only what earlier steps produced):
    population -> world (cars, homes, energy) -> banking (products, balances)
    -> events -> labels (take-up from a causal model) -> transactions
    -> digital activity -> scenario pack -> writer (warehouse tables + truth vault)
"""
