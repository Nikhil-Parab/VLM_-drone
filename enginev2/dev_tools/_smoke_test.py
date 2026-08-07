import sys
sys.path.insert(0, '.')

from voice_command_interface import ACTION_SYNONYMS, VALID_ACTIONS

test_cases = [
    ('go back',            'rtl'),
    ('return home',        'rtl'),
    ('don t move',         'hover'),
    ('chase the red car',  'follow'),
    ('orbit the building', 'scan'),
]

print('=== Synonym normalisation (no model) ===')
all_ok = True
for phrase, expected in test_cases:
    resolved = 'hover'
    phrase_lower = phrase.lower()
    for syn, canonical in sorted(ACTION_SYNONYMS.items(), key=lambda x: -len(x[0])):
        if syn in phrase_lower:
            resolved = canonical
            break
    status = 'OK  ' if resolved == expected else 'FAIL'
    if status == 'FAIL':
        all_ok = False
    print(f'  {status}  "{phrase}" -> {resolved}  (expected {expected})')

print()
print('=== VALID_ACTIONS check ===')
expected_actions = {'follow','track','hover','scan','survey','land','rtl','stop','search','return'}
missing = expected_actions - VALID_ACTIONS
extra   = VALID_ACTIONS - expected_actions
ok = not missing and not extra
print(f'  Actions OK: {ok}  missing={missing}  extra={extra}')

print()
if all_ok and ok:
    print('ALL LOGIC TESTS PASSED.')
    sys.exit(0)
else:
    print('SOME TESTS FAILED.')
    sys.exit(1)
