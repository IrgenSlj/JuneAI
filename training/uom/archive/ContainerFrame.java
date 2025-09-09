import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.event.*;
import java.util.ArrayList;
import javax.swing.*;

public class ContainerFrame extends JFrame {
    
    private JTextField codeField;
    private JTextField destinationField;    
    private JTextField weightField;
    private JTextField powerField;
    private JButton createBulkButton;
    private JButton createRefridgeratorButton;
    private JList shipList;
    private JPanel containerPanel;
    private JPanel centralPanel;
    private ArrayList<Ship> ships;


    public ContainerFrame(ArrayList<Ship> someShips) {

        ships = someShips;

        codeField = new JTextField("code");
        destinationField = new JTextField("destination");
        weightField = new JTextField("weight");
        powerField = new JTextField("power");
        createBulkButton = new JButton("Create bulk container");
        createRefridgeratorButton = new JButton("Create refridgerated container");
        shipList = new JList();
        containerPanel = new JPanel();
        centralPanel = new JPanel();

        GridLayout grid = new GridLayout(3, 2);
        containerPanel.setLayout(grid);

        containerPanel.add(codeField);
        containerPanel.add(destinationField);
        containerPanel.add(weightField);
        containerPanel.add(powerField);
        containerPanel.add(createBulkButton);
        containerPanel.add(createRefridgeratorButton);

        BorderLayout border = new BorderLayout();
        centralPanel.setLayout(border);

        centralPanel.add(shipList, BorderLayout.NORTH);
        centralPanel.add(containerPanel, BorderLayout.CENTER);

        DefaultListModel model = new DefaultListModel();

        for (Ship ship: ships) {
            model.addElement(ship.getName());
        }

        shipList.setModel(model);

        setContentPane(centralPanel);

        ButtonListener listener = new ButtonListener();
        createBulkButton.addActionListener(listener);
        createRefridgeratorButton.addActionListener(listener);

        this.setVisible(true);
        this.setTitle("Containers manager app");
        this.setSize(400, 400);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    class ButtonListener implements ActionListener {

        public void actionPerformed(ActionEvent e) {

            String code = codeField.getText();
            String destination = destinationField.getText();
            String selectedShipName = (String) shipList.getSelectedValue();

            Ship selectedShip = null;
            for (Ship ship: ships) {
                if (ship.getName().equals(selectedShipName))
                    selectedShip = ship;

            if (selectedShip != null) {
                if (e.getSource() == createBulkButton) {
                    String weightText = weightField.getText();
                    int weight = Integer.parseInt(weightText);
                    BulkContainer newContainer = new BulkContainer(code, destination, weight);
                    selectedShip.addContainer(newContainer);
                } else {
                    String powerText = powerField.getText();
                    double power = Double.parseDouble(powerText);
                    RefrigeratedContainer newContainer = new RefrigeratedContainer(code, destination, power);
                    selectedShip.addContainer(newContainer);
                }

                System.out.println("Selected ship charge: " + selectedShip.getTotalCharge());
            }
            }
        }
    }
}
