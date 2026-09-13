from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
svm_model = joblib.load("svm_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Responses
# Responses
responses = {
    "Refund_not_showing_up": "Your refund has been processed, but it may take some time to appear in your account. Please check again later.",
    
    "activate_my_card": "You can activate your card through the banking app by following the card activation instructions.",
    "age_limit": "You must meet the minimum age requirement to open an account.",
    "apple_pay_or_google_pay": "You can use your card with supported mobile payment services such as Apple Pay or Google Pay.",
    "atm_support": "You can use your card at supported ATMs to withdraw cash.",
    "automatic_top_up": "Automatic top up allows you to automatically add money to your account when your balance gets low.",
    "balance_not_updated_after_bank_transfer": "Your balance may take some time to update after a bank transfer. Please wait and check again.",
    "balance_not_updated_after_cheque_or_cash_deposit": "Cash or cheque deposits may take some time to appear in your balance. Please check again later.",
    "beneficiary_not_allowed": "This beneficiary cannot be used for the transfer. Please check the beneficiary details and try again.",
    "cancel_transfer": "If the transfer has not been completed yet, you may be able to cancel it. Please check the transfer status.",
    "card_about_to_expire": "Your card is approaching its expiration date. A replacement card should be provided according to your bank's policy.",
    "card_acceptance": "Your card can be used at merchants that support your card network.",
    "card_arrival": "Your card should arrive within the estimated delivery period shown in your account.",
    "card_delivery_estimate": "Card delivery usually takes a few business days. Check your app for the latest delivery estimate.",
    "card_linking": "You can link your card through the banking app. Make sure the card details are entered correctly.",
    "card_not_working": "Your card may be temporarily blocked or there may be an issue with the card. Please check its status in the app.",
    "card_payment_fee_charged": "A fee may have been charged for the card payment. Please check the transaction details for more information.",
    "card_payment_not_recognised": "I'm sorry about that. This payment may not have been made by you. Please check the transaction details and contact support if you still don't recognize it.",
    "card_payment_wrong_exchange_rate": "The exchange rate used for this card payment may differ from the expected rate. Please check the transaction details.",
    "card_swallowed": "If an ATM has kept your card, contact your bank or the ATM provider as soon as possible.",
    "cash_withdrawal_charge": "A fee may have been charged for the cash withdrawal. Please check the transaction details.",
    "cash_withdrawal_not_recognised": "If you don't recognize this cash withdrawal, check the transaction details and contact support immediately.",
    "change_pin": "You can change your PIN through the banking app or at a supported ATM.",
    "compromised_card": "If you believe your card has been compromised, freeze it immediately and contact support.",
    "contactless_not_working": "If contactless payments are not working, try another payment method and check whether your card's contactless feature is enabled.",
    "country_support": "Please check whether your country is supported for this banking service.",
    "declined_card_payment": "Your card payment was declined. Please check your card status, available balance, and payment details.",
    "declined_cash_withdrawal": "Your cash withdrawal was declined. Check your available balance and try another supported ATM.",
    "declined_transfer": "Your transfer was declined. Please check the transfer details and try again.",
    "direct_debit_payment_not_recognised": "I don't recognize this direct debit payment. Please review the transaction and contact support if necessary.",
    "disposable_card_limits": "Disposable virtual cards may have usage or transaction limits. Please check the limits in your banking app.",
    "edit_personal_details": "You can update your personal details through the banking app if this option is available for your account.",
    "exchange_charge": "Currency exchange may include a fee depending on the transaction and currency.",
    "exchange_rate": "The exchange rate is based on the current rate used by the service and may change over time.",
    "exchange_via_app": "You can exchange supported currencies directly through the banking app.",
    "extra_charge_on_statement": "An extra charge on your statement may be a small verification or additional transaction charge. Check the transaction details.",
    "failed_transfer": "The transfer failed. Please check the recipient details, your balance, and try again.",
    "fiat_currency_support": "The service supports a range of fiat currencies. Check the app for the currently supported currencies.",
    "get_disposable_virtual_card": "You can create a disposable virtual card through the banking app if your account supports this feature.",
    "get_physical_card": "You can request a physical card through the banking app.",
    "getting_spare_card": "You may be able to request an additional spare card through the banking app.",
    "getting_virtual_card": "You can get a virtual card through the banking app if your account is eligible.",
    "lost_or_stolen_card": "If your card is lost or stolen, freeze it immediately and contact support to request a replacement.",
    "lost_or_stolen_phone": "If your phone is lost or stolen, secure your account immediately and contact support.",
    "order_physical_card": "You can order a physical card through the banking app.",
    "passcode_forgotten": "If you forgot your passcode, use the recovery option in the app to regain access to your account.",
    "pending_card_payment": "Your card payment is still pending. Please wait for the transaction to finish.",
    "pending_cash_withdrawal": "Your cash withdrawal is still pending. Please wait for the transaction to be completed.",
    "pending_top_up": "Your top up is still pending. Please wait for the transaction to finish.",
    "pending_transfer": "Your transfer is still pending. Please wait for the transaction to be completed.",
    "pin_blocked": "Your PIN has been blocked. Please follow the instructions in the app to unblock or reset it.",
    "receiving_money": "You can receive money through supported transfer methods. Check the app for the available options.",
    "request_refund": "You can request a refund for an eligible transaction through the appropriate support or refund process.",
    "reverted_card_payment?": "The card payment was reversed. The money should normally return to your account after the reversal is completed.",
    "supported_cards_and_currencies": "The service supports specific card types and currencies. Please check the supported options in the app.",
    "terminate_account": "If you want to close your account, follow the account closure process in the app or contact support.",
    "top_up_by_bank_transfer_charge": "Bank transfer top ups may have fees depending on the transfer method.",
    "top_up_by_card_charge": "A fee may apply when topping up with a card depending on the card and transaction.",
    "top_up_by_cash_or_cheque": "Cash or cheque top ups may not be supported. Please check the available top up methods.",
    "top_up_failed": "Your top up failed. Please check your card details, available balance, and try again.",
    "top_up_limits": "Top ups may have daily or transaction limits. Check the app for your current limits.",
    "top_up_reverted": "Your top up was reverted. The money should normally return to the original payment method.",
    "topping_up_by_card": "You can top up your account using a supported bank card through the banking app.",
    "transaction_charged_twice": "It looks like the same transaction may have been charged twice. Please check both transactions and contact support if needed.",
    "transfer_fee_charged": "A fee may have been charged for the transfer. Please check the transaction details.",
    "transfer_into_account": "You can add money to your account using supported bank transfer methods.",
    "transfer_not_received_by_recipient": "The recipient has not received the transfer yet. Please check the transfer status and recipient details.",
    "transfer_timing": "Transfer times depend on the transfer method and destination. Some transfers may take several business days.",
    "unable_to_verify_identity": "We couldn't verify your identity. Please make sure your documents and personal information are clear and correct.",
    "verify_my_identity": "Identity verification is required to confirm your account. Follow the verification steps in the app.",
    "verify_source_of_funds": "You may be asked to provide information or documents showing where your money comes from.",
    "verify_top_up": "Your top up may require verification for security reasons. Follow the instructions shown in the app.",
    "virtual_card_not_working": "Your virtual card may be temporarily unavailable or restricted. Check its status in the app.",
    "visa_or_mastercard": "Your card may be issued as either Visa or Mastercard depending on the card provided to you.",
    "why_verify_identity": "Identity verification helps protect your account and comply with financial security requirements.",
    "wrong_amount_of_cash_received": "The amount of cash you received does not match the amount shown in the transaction. Please check the transaction details and contact support.",
    "wrong_exchange_rate_for_cash_withdrawal": "The exchange rate used for your cash withdrawal may be different from what you expected. Please check the transaction details."
}


def predict_intent(text):
    text = tfidf.transform([text])
    prediction = svm_model.predict(text)
    return prediction[0]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "error": "Please enter a message."
        }), 400

    intent = predict_intent(text)
    response = responses.get(
    intent,
    "I'm sorry, I couldn't find a response for this issue. Please contact support."
)

    return jsonify({
        "intent": intent,
        "response": response
    })


if __name__ == "__main__":
    app.run(debug=True)