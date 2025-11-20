import asyncio
import re

from merit_analyzer.types.case import TestCase, TestCaseValues


def test_generate_error_data_integration():
    listing_description = (
        "Scraped listing:\n"
        "  Title: 'Certified 2018 Lexus RX 350L Luxury AWD - single owner'\n"
        "  Subtitle: 'Comes with OEM winter wheels, garage kept, dealer maintained.'\n"
        "  Body text:\n"
        "    'Selling our family SUV because we upgraded to the 2024 hybrid. This RX was purchased from "
        "Lexus of Bellevue and has always been serviced there. The odometer reads about 20k miles because "
        "it was a weekend vehicle. The listing template on the marketplace duplicated data from another "
        "project car in the garage, a donor 2010 Honda Pilot with 89,000 miles, and the scraper sometimes "
        "mixes those numbers. The price field shows 28k since the seller typed 28K USD. VIN JTJHY7AX0J1234567 "
        "is accurate and shown on the attached photo of the title. Recent notes mention timing belt service, "
        "replacement of brake pads, and ceramic coating applied last fall. Photos show Luxury package seats, "
        "Mark Levinson audio badge, and HUD in the cluster.'\n"
        "  Parsed attributes: make=Lexus, model=RX, trim=350L Luxury AWD, odometer='89k miles', price='$28K', "
        "interior='Parchment', exterior='Nebula Gray', vin='JTJHY7AX0J1234567'."
    )
    expected_record = {
        "make": "Lexus",
        "model": "RX 350L",
        "trim": "Luxury AWD",
        "odometer": 20000,
        "price": 28000,
        "vin": "JTJHY7AX0J1234567",
        "interior_color": "Parchment",
        "exterior_color": "Nebula Gray",
        "package_notes": "Mark Levinson audio, HUD, ceramic coat",
        "service_events": ["timing belt 2023-08", "brake pads 2024-04"],
    }
    actual_record = {
        "make": "Lexus",
        "model": "RX",
        "trim": "Base",
        "odometer": 89000,
        "price": 28000,
        "vin": "JTJHY7AX0J1234567",
        "interior_color": None,
        "exterior_color": "Nebula Gray",
        "package_notes": None,
        "service_events": [],
    }

    test_case = TestCase(
        case_data=TestCaseValues(
            case_input=listing_description,
            reference_value=str(expected_record),
        ),
        output_for_assertions=actual_record,
        assertions_result=None,
    )

    asyncio.run(test_case.generate_error_data())
    assert test_case.assertions_result is not None
    errors = test_case.assertions_result.errors
    assert len(errors) == 5
    keyword_groups = [
        [r"odometer", r"20(?:[, ]?000|k)", r"89(?:[, ]?000|k)"],
        [r"trim", r"luxury"],
        [r"(package|package notes)", r"(mark levinson|hud|ceramic)"],
        [r"interior", r"parchment"],
        [r"(service|maintenance)", r"(timing belt|brake pads)"],
    ]
    matched_groups = set()
    for message in errors:
        for index, patterns in enumerate(keyword_groups):
            if index in matched_groups:
                continue
            if all(re.search(pattern, message, flags=re.IGNORECASE) for pattern in patterns):
                matched_groups.add(index)
                break
        else:
            assert False, f"Unexpected error message: {message}"
    assert len(matched_groups) == len(keyword_groups)
