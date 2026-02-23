Idea is to clean test_models.py and general logic around pytest.
Some key improvments needed are:
- record_properties should for the fields known before the test should be written before hand.
    - this will enable saving partial tags in the event that test runned with `--forked` aborts
- record_properties that are calculated during test should be record in test teardown.
- remove try catch as pytest already internaly has that mechanism.
- enable xpass to work properly 

Do you have any other ideas what can be done?