library(shiny)
library(dplyr)
library(ggplot2)

project_dir <- normalizePath(".", winslash = "/", mustWork = TRUE)
config_dir <- file.path(project_dir, "config")
output_dir <- file.path(project_dir, "output")
sessions_dir <- file.path(output_dir, "sessions")

doctors_path <- file.path(config_dir, "doctors.csv")
lectures_path <- file.path(config_dir, "lectures.csv")
attendance_sample_path <- file.path(sessions_dir, "test_summary_attendance.csv")
summary_sample_path <- file.path(sessions_dir, "test_summary_summary.csv")

read_safe_csv <- function(path) {
  if (!file.exists(path)) return(data.frame())
  read.csv(path, stringsAsFactors = FALSE)
}

doctors <- read_safe_csv(doctors_path)
lectures <- read_safe_csv(lectures_path)

ui <- fluidPage(
  titlePanel("AI Classroom Monitoring - Shiny Dashboard"),
  uiOutput("main_ui")
)

server <- function(input, output, session) {
  rv <- reactiveValues(
    logged_in = FALSE,
    doctor_id = "",
    doctor_name = "",
    session_result = ""
  )

  output$main_ui <- renderUI({
    if (!rv$logged_in) {
      fluidRow(
        column(
          width = 4, offset = 4,
          wellPanel(
            h3("Doctor Login"),
            textInput("username", "Username"),
            passwordInput("password", "Password"),
            actionButton("login_btn", "Login", class = "btn-primary"),
            br(), br(),
            textOutput("login_msg")
          )
        )
      )
    } else {
      navbarPage(
        title = paste("Welcome,", rv$doctor_name),
        tabPanel(
          "Session Control",
          fluidRow(
            column(
              width = 4,
              selectInput("subject_id", "Subject", choices = NULL),
              selectInput("lecture_id", "Lecture", choices = NULL),
              textInput("session_id", "Session ID", value = paste0("lecture_", format(Sys.time(), "%Y%m%d_%H%M%S"))),
              numericInput("camera_index", "Camera Index", value = 0, min = 0),
              selectInput("camera_backend", "Camera Backend", choices = c("auto", "dshow", "msmf"), selected = "dshow"),
              numericInput("tolerance", "Tolerance", value = 0.65, min = 0.3, max = 0.9, step = 0.01),
              numericInput("max_frames", "Max Frames (for demo)", value = 400, min = 0),
              checkboxInput("show_window", "Show Camera Window", value = TRUE),
              checkboxInput("enable_emotion", "Enable Emotion Detection", value = TRUE),
              selectInput("emotion_engine", "Emotion Engine", choices = c("heuristic", "auto", "deepface"), selected = "heuristic"),
              numericInput("emotion_min_face_size", "Emotion Min Face Size", value = 48, min = 24, max = 256),
              actionButton("start_session", "Start Session", class = "btn-success")
            ),
            column(
              width = 8,
              h4("Session Output"),
              verbatimTextOutput("session_output"),
              helpText("Press 'q' in OpenCV window to stop if show window is enabled in CLI mode.")
            )
          )
        ),
        tabPanel(
          "Attendance",
          fluidRow(
            column(width = 12, tableOutput("attendance_table"))
          )
        ),
        tabPanel(
          "Statistics",
          fluidRow(
            column(width = 6, tableOutput("summary_table")),
            column(width = 6, plotOutput("status_plot", height = "320px"))
          ),
          fluidRow(
            column(width = 12, plotOutput("attitude_plot", height = "320px"))
          )
        )
      )
    }
  })

  observeEvent(input$login_btn, {
    req(nrow(doctors) > 0)
    user <- trimws(input$username)
    pass <- trimws(input$password)
    row <- doctors %>% filter(username == user, password == pass)
    if (nrow(row) == 1) {
      rv$logged_in <- TRUE
      rv$doctor_id <- as.character(row$doctor_id[1])
      rv$doctor_name <- as.character(row$doctor_name[1])
    }
  })

  output$login_msg <- renderText({
    if (input$login_btn < 1) return("")
    user <- trimws(input$username)
    pass <- trimws(input$password)
    ok <- nrow(doctors %>% filter(username == user, password == pass)) == 1
    if (ok) "Login successful." else "Invalid username or password."
  })

  observe({
    req(rv$logged_in)
    doc_lectures <- lectures %>% filter(doctor_id == rv$doctor_id)
    subjects <- sort(unique(doc_lectures$subject_id))
    updateSelectInput(session, "subject_id", choices = subjects, selected = subjects[1])
  })

  observe({
    req(rv$logged_in, input$subject_id)
    subject_lectures <- lectures %>% filter(doctor_id == rv$doctor_id, subject_id == input$subject_id)
    if (nrow(subject_lectures) == 0) {
      updateSelectInput(session, "lecture_id", choices = character(0))
    } else {
      labels <- paste(subject_lectures$lecture_id, "-", subject_lectures$day_name, subject_lectures$start_time)
      choices <- setNames(subject_lectures$lecture_id, labels)
      updateSelectInput(session, "lecture_id", choices = choices, selected = subject_lectures$lecture_id[1])
    }
  })

  observeEvent(input$start_session, {
    req(input$lecture_id, input$session_id)
    python_cmd <- "python"
    script_path <- file.path(project_dir, "take_attendance_realtime.py")
    args <- c(
      shQuote(script_path),
      "--embeddings", shQuote(file.path(output_dir, "face_embeddings.npz")),
      "--source", as.character(input$camera_index),
      "--camera-backend", as.character(input$camera_backend),
      "--output-dir", shQuote(sessions_dir),
      "--session-id", shQuote(input$session_id),
      "--tolerance", as.character(input$tolerance),
      "--frame-skip", "1",
      "--max-frames", as.character(input$max_frames)
    )
    if (isTRUE(input$enable_emotion)) {
      args <- c(
        args,
        "--enable-emotion",
        "--emotion-engine", as.character(input$emotion_engine),
        "--emotion-min-face-size", as.character(input$emotion_min_face_size)
      )
    }
    if (isTRUE(input$show_window)) {
      args <- c(args, "--show-window")
    }

    result <- tryCatch({
      out <- system2(command = python_cmd, args = args, stdout = TRUE, stderr = TRUE)
      paste(out, collapse = "\n")
    }, error = function(e) {
      paste("Session run failed:", conditionMessage(e))
    })
    rv$session_result <- result
  })

  output$session_output <- renderText({
    if (!nzchar(rv$session_result)) return("No session started yet.")
    rv$session_result
  })

  output$attendance_table <- renderTable({
    req(rv$logged_in)
    p <- file.path(sessions_dir, paste0(input$session_id, "_attendance.csv"))
    if (!file.exists(p)) p <- attendance_sample_path
    df <- read_safe_csv(p)
    if (nrow(df) == 0) return(data.frame(Message = "No attendance data yet."))
    head(df, 50)
  })

  output$summary_table <- renderTable({
    req(rv$logged_in)
    p <- file.path(sessions_dir, paste0(input$session_id, "_summary.csv"))
    if (!file.exists(p)) p <- summary_sample_path
    df <- read_safe_csv(p)
    if (nrow(df) == 0) return(data.frame(Message = "No summary data yet."))
    df
  })

  output$status_plot <- renderPlot({
    req(rv$logged_in)
    p <- file.path(sessions_dir, paste0(input$session_id, "_summary.csv"))
    if (!file.exists(p)) p <- summary_sample_path
    df <- read_safe_csv(p)
    if (nrow(df) == 0) return(NULL)
    plot_df <- data.frame(
      status = c("present", "late", "left_early", "absent"),
      count = c(df$present_count[1], df$late_count[1], df$left_early_count[1], df$absent_count[1])
    )
    ggplot(plot_df, aes(x = status, y = count, fill = status)) +
      geom_col() +
      theme_minimal(base_size = 13) +
      labs(title = "Session Status Distribution", x = "Status", y = "Count") +
      guides(fill = "none")
  })

  output$attitude_plot <- renderPlot({
    req(rv$logged_in)
    p <- file.path(sessions_dir, paste0(input$session_id, "_summary.csv"))
    if (!file.exists(p)) p <- summary_sample_path
    df <- read_safe_csv(p)
    if (nrow(df) == 0) return(NULL)
    if (!all(c("engaged_count", "normal_count", "at_risk_count") %in% names(df))) return(NULL)
    plot_df <- data.frame(
      attitude = c("engaged", "normal", "at_risk"),
      count = c(df$engaged_count[1], df$normal_count[1], df$at_risk_count[1])
    )
    ggplot(plot_df, aes(x = attitude, y = count, fill = attitude)) +
      geom_col() +
      theme_minimal(base_size = 13) +
      labs(title = "Attitude Distribution", x = "Attitude", y = "Count") +
      guides(fill = "none")
  })
}

shinyApp(ui = ui, server = server)
