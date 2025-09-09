public class BankAccount {
    
    private double balance;
    private String code;

    public BankAccount(String code, double balance) {
        this.code = code;
        this.balance = balance;
    }

    public double getBalance() {
        return balance;
    }

    public String getCode() {
        return code;
    }

    public int hashCode() {
        return code.hashCode();
    }

    public boolean equals(Object other) {
        BankAccount otherAccount = (BankAccount) other;
        return otherAccount.getCode().equals(this.getCode());
    }
}
