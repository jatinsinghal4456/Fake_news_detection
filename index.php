<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = $_POST['username'];
    $password = $_POST['password'];

    // Validate the username and password
    if ($username === "admin" && $password === "password") {
        // Redirect to the dashboard or home page
        header("Location: dashboard.php");
        exit();
    } else {
        // Display an error message
        echo "Invalid username or password!";
    }
}
?>
