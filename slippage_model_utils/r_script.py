# test_variable_creator.py
from r_script_wrapper import VariableCreator


def test_basic_text_preproc():
    """Test the basic_text_preproc function"""
    creator = VariableCreator()

    # Test case 1: Simple text
    original = "Mr. John Doe Inc. Box 123"
    print(f"\nTest 1: '{original}'")
    try:
        cleaned = creator.basic_text_preproc(original)
        print(f"Result: '{cleaned}'")
    except Exception as e:
        print(f"Error: {e}")

    # Test case 2: Multiple patterns
    original2 = "ABC Corporation Ltd. 456"
    print(f"\nTest 2: '{original2}'")
    try:
        cleaned2 = creator.basic_text_preproc(original2)
        print(f"Result: '{cleaned2}'")
    except Exception as e:
        print(f"Error: {e}")

    # Test case 3: Missing value
    original3 = "Not Selected"
    print(f"\nTest 3: '{original3}'")
    try:
        cleaned3 = creator.basic_text_preproc(original3)
        print(f"Result: '{cleaned3}'")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_basic_text_preproc()
